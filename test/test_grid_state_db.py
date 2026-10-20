"""Tail-of-run correctness for the multi-node grid search.

These cover the failure that let a grid finish "complete" with jobs unwritten:
nodes retired the moment their local queue drained, stranding claims nobody
picked up, and compaction then wrote the completion flag over the resulting gap
and deleted the batch files that proved it.

Everything here is DB- and file-level, so it runs in a second without a cluster,
a dataset, or a training step.
"""
import sqlite3
import time
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from project.parallel_grid_search.code.parallel_utils import (
    COMPACT_SENTINEL,
    GridSearchDB,
    JobGenerator,
    JobInterface,
    ResourceLimits,
    STATUS_CLAIMED,
    STATUS_DONE,
    STATUS_FAILED,
    STATUS_PENDING,
)
from project.parallel_grid_search.code.train_model_parallel import (
    ResourceAwareScheduler,
    _GridSearchProgress,
    _compact_results,
    _missing_result_pairs,
    _read_batch_files,
)


class _NullJob(JobInterface):
    """Smallest thing JobGenerator will hand out; never actually run here."""

    def _run(self, device):
        return {}


def _factory(i, j, total_configs, total_samples, locks):
    return _NullJob(i, j, total_configs, total_samples, locks)


@pytest.fixture
def db(tmp_path):
    d = GridSearchDB.open(tmp_path / 'state.db', total_configs=2, samples_per_config=3)
    yield d
    d.close()


def _status_of(db_path, i, j):
    with sqlite3.connect(db_path) as conn:
        return conn.execute('SELECT status FROM jobs WHERE i=? AND j=?', (i, j)).fetchone()[0]


def _write_parquet(path, pairs):
    pq.write_table(
        pa.table({'i': [i for i, _ in pairs], 'j': [j for _, j in pairs]}), path
    )


# --------------------------------------------------------------------------
# status bookkeeping
# --------------------------------------------------------------------------

def test_failed_is_terminal_but_distinct_from_done(db, tmp_path):
    """A failed job must not count as unfinished, and must not masquerade as done.

    The old code marked give-ups as 'done', which is what made an unexplained gap
    indistinguishable from an explained one at compaction time.
    """
    db.claim_next_batch(6)
    db.mark_done_batch([(0, 0), (0, 1)])
    db.mark_failed([(1, 2)])

    counts = db.counts()
    assert counts.get('done') == 2
    assert counts.get('failed') == 1
    assert _status_of(tmp_path / 'state.db', 1, 2) == STATUS_FAILED

    # 6 total, 2 done, 1 failed -> 3 still to run
    assert db.unfinished_count() == 3


def test_count_live_nodes_counts_held_work_not_registry_rows(tmp_path):
    """Heartbeat rows outlive their process, one per node per run, so counting them
    overstates an 8-node crew several times over and the ETA reads far too
    optimistic.  Held claims track only the nodes actually consuming work."""
    db = GridSearchDB.open(tmp_path / 'state.db', 2, 3, node_id='observer')
    for ghost in ('hpc1_old', 'hpc2_older', 'hpc3_ancient'):
        _beat(db, ghost, time.time() - 50_000)

    assert db.count_live_nodes() == 1, "ghosts hold no work; never reports zero either"

    peer = GridSearchDB.open(tmp_path / 'state.db', 2, 3, node_id='peer')
    peer.claim_next_batch(3)
    db.claim_next_batch(3)

    assert db.count_live_nodes() == 2, "exactly the two nodes holding claims"
    peer.close()


def test_the_compaction_sentinel_is_never_a_node_nor_swept_away(db):
    """The sentinel shares the heartbeats table but claims no jobs, and it never
    beats — so the sweep must not mistake it for a dead node and delete it, which
    would let a second node start compacting on top of the first."""
    db.claim_next_batch(6)
    assert db.try_claim_compaction()

    assert db.count_live_nodes() == 1, "sentinel is not a node"
    db.release_dead_holders(timeout_s=0)
    db.release_dead_holders(timeout_s=0)
    assert COMPACT_SENTINEL in db._read_heartbeats()


def test_unfinished_count_reaches_zero_only_when_nothing_is_left(db):
    assert db.unfinished_count() == 6
    db.claim_next_batch(6)
    assert db.unfinished_count() == 6, "claimed jobs are still unfinished work"
    db.mark_done_batch([(i, j) for i in range(2) for j in range(3)])
    assert db.unfinished_count() == 0


# --------------------------------------------------------------------------
# releasing claims — the "stranded work" half of the bug
# --------------------------------------------------------------------------

def test_reset_to_pending_only_touches_claimed(db, tmp_path):
    db.claim_next_batch(6)
    db.mark_done_batch([(0, 0)])
    db.mark_failed([(0, 1)])

    n = db.reset_to_pending([(0, 0), (0, 1), (0, 2)])

    assert n == 1, "only the still-claimed job may be handed back"
    assert _status_of(tmp_path / 'state.db', 0, 0) == STATUS_DONE
    assert _status_of(tmp_path / 'state.db', 0, 1) == STATUS_FAILED
    assert _status_of(tmp_path / 'state.db', 0, 2) == STATUS_PENDING


def test_release_buffered_hands_back_undispatched_claims(tmp_path):
    """A departing node must not take its unstarted claims to the grave.

    Peers read leftover claims as somebody else's live work and decline to
    compact, so the grid never gets flagged complete.
    """
    db = GridSearchDB.open(tmp_path / 'state.db', total_configs=2, samples_per_config=3)
    gen = JobGenerator(job_factory=_factory, total_configs=2, samples_per_config=3, db=db)

    it = iter(gen)
    next(it)  # claims a whole batch, yields one job from it

    assert db.counts().get('claimed') == 6
    freed = gen.release_buffered()

    assert freed == 5, "every claimed-but-unyielded job goes back"
    assert db.counts().get('pending') == 5
    assert gen.release_buffered() == 0, "second call is a no-op"
    db.close()


def test_release_buffered_is_safe_without_a_db(tmp_path):
    gen = JobGenerator(job_factory=_factory, total_configs=2, samples_per_config=3)
    assert gen.release_buffered() == 0


# --------------------------------------------------------------------------
# crash recovery and the heartbeats table
#
# A holder is judged by whether its heartbeat *changes* while another node watches,
# timed on the watcher's own monotonic clock.  No two machines' timestamps are ever
# compared, so a skewed clock (hpc9, ~79 min fast) cannot make a live node look dead.
# --------------------------------------------------------------------------

def _beat(db, node_id, value):
    """Force a heartbeat timestamp, standing in for a peer this process can't run."""
    with db._tx() as conn:
        conn.execute("INSERT OR REPLACE INTO heartbeats (node_id, last_seen) VALUES (?, ?)",
                     (node_id, value))


def _two_nodes(tmp_path):
    """An observer and a peer on one state.db, holding three jobs each."""
    observer = GridSearchDB.open(tmp_path / 'state.db', 2, 3, node_id='observer')
    peer = GridSearchDB.open(tmp_path / 'state.db', 2, 3, node_id='peer')
    peer.claim_next_batch(3)
    observer.claim_next_batch(3)
    return observer, peer


def test_a_holder_whose_beat_changes_is_alive_and_one_that_never_does_is_dead(tmp_path):
    observer, peer = _two_nodes(tmp_path)
    assert observer.watch_claim_holders() == {'peer': 'unknown'}, "one sighting proves nothing"
    _beat(observer, 'peer', time.time() + 1)
    assert observer.watch_claim_holders() == {'peer': 'alive'}
    assert observer.watch_claim_holders(timeout_s=0) == {'peer': 'dead'}, "frozen past the timeout"
    peer.close()


def test_a_crashed_peer_has_its_jobs_reclaimed(tmp_path):
    """A crashed node's claims must come back, or the grid can never finish them."""
    observer, peer = _two_nodes(tmp_path)
    assert observer.release_dead_holders(timeout_s=0)[0] == 0, "first look: not judged yet"
    assert observer.release_dead_holders(timeout_s=0)[0] == 3, "its beat stood still"
    assert observer.counts() == {'pending': 3, 'claimed': 3}, "the observer keeps its own"
    assert 'peer' not in observer._read_heartbeats(), "a dead node's row is dropped"
    assert 'observer' in observer._read_heartbeats(), "a live node's row survives"
    peer.close()


def test_a_beating_peer_keeps_its_work(tmp_path):
    observer, peer = _two_nodes(tmp_path)
    observer.release_dead_holders()
    _beat(observer, 'peer', time.time() + 60)
    assert observer.release_dead_holders() == (0, {'peer': 'alive'})
    assert observer.counts().get('claimed') == 6
    peer.close()


def test_a_skewed_clock_cannot_make_a_live_peer_look_dead(tmp_path):
    """The hpc9 failure, from both sides: a beat stamped 79 min off is still a beat."""
    observer, peer = _two_nodes(tmp_path)
    for skew in (-79 * 60, 79 * 60):
        _beat(observer, 'peer', time.time() + skew)
        observer.watch_claim_holders()
        _beat(observer, 'peer', time.time() + skew + 60)
        assert observer.release_dead_holders() == (0, {'peer': 'alive'})
    peer.close()


def test_a_clean_exit_hands_its_claims_back_at_once(tmp_path):
    """close() drops the heartbeat row, so a graceful shutdown needs no watching."""
    observer, peer = _two_nodes(tmp_path)
    peer.close()
    assert observer.release_dead_holders()[0] == 3, "no row, no wait"


def test_a_node_joining_a_config_reclaims_only_the_leavers(tmp_path):
    """Probe 2, 2026-10-13: hpc9 opened each next config's state.db and reset every live
    peer's claims (c4: 254 jobs run twice).  On first look only a holder without a
    heartbeat row is dead; one with a row, however old it looks, gets watched first."""
    observer, peer = _two_nodes(tmp_path)
    _beat(observer, 'peer', time.time() - 78 * 60)   # alive, but reads as dead to a fast clock
    leaver = GridSearchDB.open(tmp_path / 'state.db', 3, 3, node_id='leaver')
    leaver.claim_next_batch(3)
    leaver.close()

    assert observer.release_dead_holders()[0] == 3, "the leaver's claims come back"
    assert observer.counts().get('claimed') == 6, "the old-looking peer keeps its work"
    assert 'peer' in observer._read_heartbeats(), "and its row is not dropped"
    peer.close()


def test_release_claims_of_frees_only_the_named_holders(tmp_path):
    observer, peer = _two_nodes(tmp_path)
    assert observer.release_claims_of(['peer']) == 3
    assert observer.counts() == {'pending': 3, 'claimed': 3}
    peer.close()


def test_ownerless_claims_are_released_too(tmp_path):
    observer, peer = _two_nodes(tmp_path)
    with observer._tx() as conn:
        conn.execute("UPDATE jobs SET node_id=NULL WHERE node_id='peer'")
    assert observer.release_dead_holders() == (3, {None: 'dead'})
    peer.close()


# --------------------------------------------------------------------------
# the tail: when a node with nothing left to claim may leave
#
# Final run 2026-10-16: 14 nodes idled ~10 min each on 943 "unfinished" jobs that
# were finished results hpc4 had not flushed yet.  A drained node now leaves once
# every outstanding claim has a live holder, and stays only to take over a dead one's.
# --------------------------------------------------------------------------

def _idle(db):
    """The state _ready_to_exit reads, without a resource manager or worker pools."""
    return SimpleNamespace(db=db, REARM_DELAY_S=0.0, _last_tail_check=0.0, _tail_logged=False)


def _ready_to_exit(node):
    return ResourceAwareScheduler._ready_to_exit(node)


def test_an_idle_node_leaves_once_every_claim_has_a_live_holder(tmp_path):
    observer, peer = _two_nodes(tmp_path)
    observer.claim_next_batch(6)                       # nothing pending any more
    node = _idle(observer)
    assert not _ready_to_exit(node), "first sighting: the holder may be dead"
    _beat(observer, 'peer', time.time() + 1)
    assert _ready_to_exit(node), "its beat moved: it will finish and compact without us"
    peer.close()


def test_an_idle_node_takes_over_a_dead_holders_work(tmp_path):
    observer, peer = _two_nodes(tmp_path)
    observer.claim_next_batch(6)
    peer.remove_heartbeat()                            # crashed without releasing
    assert not _ready_to_exit(_idle(observer))
    assert observer.counts().get('pending') == 3, "released for this node to re-claim"
    peer.close()


def test_an_idle_node_stays_while_work_is_pending(tmp_path):
    observer, peer = _two_nodes(tmp_path)
    observer.release_claims_of(['peer'])
    assert not _ready_to_exit(_idle(observer))
    peer.close()


def test_an_idle_node_does_not_wait_on_its_own_claims(tmp_path):
    """Its own claims are results about to be flushed, not a peer to wait for."""
    db = GridSearchDB.open(tmp_path / 'state.db', 2, 3, node_id='observer')
    db.claim_next_batch(6)
    assert _ready_to_exit(_idle(db))
    db.close()


def _feeder(t_job):
    """The state _prefetch reads: a router that last timed a job at t_job (None = never)."""
    times = {} if t_job is None else {'cpu': t_job}
    return SimpleNamespace(resource_manager=SimpleNamespace(
        bee_router=SimpleNamespace(_last_time=times), limits=ResourceLimits()))


@pytest.mark.parametrize('t_job, limit, expected', [
    (None, 32, 1),    # untimed: assume slow rather than hoard
    (195.0, 32, 5),   # the final run's jobs: 30 s of 32 workers' throughput
    (1.0, 64, 128),   # fast jobs keep the old 2 x concurrency
    (195.0, 0, 1),    # a pool backed off to nothing still gets one
])
def test_prefetch_holds_about_prefetch_s_of_work(t_job, limit, expected):
    pool = SimpleNamespace(device='cpu', current_concurrency_limit=limit)
    assert ResourceAwareScheduler._prefetch(_feeder(t_job), pool) == expected


def test_the_generator_claims_no_more_than_its_claim_batch(tmp_path):
    """The buffer is claimed work nobody else can run; it must stay as small as asked."""
    db = GridSearchDB.open(tmp_path / 'state.db', 2, 3, node_id='n')
    gen = JobGenerator(job_factory=_factory, total_configs=2, samples_per_config=3,
                       db=db, claim_batch=2)
    next(iter(gen))
    assert db.counts() == {'pending': 4, 'claimed': 2}
    db.close()


# --------------------------------------------------------------------------
# repairing a damaged grid — the "unexplained gap" half
# --------------------------------------------------------------------------

def test_reset_missing_results_only_revives_done_jobs(db, tmp_path):
    """Repair must not resurrect a failed job or steal a live claim."""
    db.claim_next_batch(6)
    db.mark_done_batch([(0, 0), (0, 1)])
    db.mark_failed([(1, 0)])

    n = db.reset_missing_results([(0, 0), (1, 0), (1, 1)])

    assert n == 1
    assert _status_of(tmp_path / 'state.db', 0, 0) == STATUS_PENDING
    assert _status_of(tmp_path / 'state.db', 1, 0) == STATUS_FAILED, "failed stays failed"
    assert _status_of(tmp_path / 'state.db', 1, 1) == STATUS_CLAIMED, "live claim untouched"


def test_missing_result_pairs_spans_compacted_and_batch_files(tmp_path):
    data_dir = tmp_path / 'data'
    data_dir.mkdir()
    _write_parquet(data_dir / 'data.parquet', [(0, 0), (0, 1)])
    _write_parquet(data_dir / 'node_000001.parquet', [(0, 2), (1, 0)])

    missing = _missing_result_pairs(data_dir, total_configs=2, samples_per_config=3)

    assert missing == {(1, 1), (1, 2)}


def test_missing_result_pairs_counts_unreadable_rows_as_missing(tmp_path):
    """An unreadable batch file is unwritten results — those jobs must re-run."""
    data_dir = tmp_path / 'data'
    data_dir.mkdir()
    _write_parquet(data_dir / 'good.parquet', [(0, 0), (0, 1), (0, 2)])
    (data_dir / 'bad.parquet').write_bytes(b'not a parquet file')

    missing = _missing_result_pairs(data_dir, total_configs=2, samples_per_config=3)

    assert missing == {(1, 0), (1, 1), (1, 2)}


# --------------------------------------------------------------------------
# compaction reporting
# --------------------------------------------------------------------------

def test_read_batch_files_reports_unreadable_instead_of_swallowing(tmp_path):
    data_dir = tmp_path / 'data'
    data_dir.mkdir()
    _write_parquet(data_dir / 'good.parquet', [(0, 0)])
    (data_dir / 'bad.parquet').write_bytes(b'garbage')

    tables, pq_files, unreadable = _read_batch_files(data_dir)

    assert len(tables) == 1
    assert len(pq_files) == 2
    assert [f.name for f in unreadable] == ['bad.parquet']


def test_compact_results_returns_row_count_and_unreadable(tmp_path):
    data_dir = tmp_path / 'data'
    data_dir.mkdir()
    _write_parquet(data_dir / 'a.parquet', [(0, 0), (0, 1)])
    _write_parquet(data_dir / 'b.parquet', [(0, 1), (0, 2)])  # overlaps: dedup to 3

    n_written, unreadable = _compact_results(data_dir)

    assert n_written == 3, "rows are unique on (i,j)"
    assert unreadable == []
    assert (data_dir / 'data.parquet').exists()


# --------------------------------------------------------------------------
# progress reporting
# --------------------------------------------------------------------------

def test_eta_accounts_for_every_node_not_just_this_one():
    """A node doing 1/10th of the work must not report a 10x ETA.

    The original maths divided global remaining work by this node's own rate,
    which on the 8-node PI run advertised ~164 h for what was really ~21 h.
    """
    pbar = _GridSearchProgress(total=1000, initial=0)
    try:
        # This node cleared 10 jobs in 100 s (0.1 j/s); 10 nodes are doing the
        # same, so the grid runs at 1.0 j/s and the remaining 900 need 15:00.
        pbar.update(global_done=100, node_done=10, elapsed_s=100.0, n_nodes=10)
        postfix = pbar._pbar.postfix
    finally:
        pbar._pbar.close()

    assert 'eta=15:00' in postfix, postfix
    assert 'nodes=10' in postfix, postfix
    assert 'this_node=10' in postfix, postfix


def test_eta_does_not_depend_on_results_having_been_flushed_yet():
    """The killer case: nodes buffer 1000 results before writing, so the global
    'done' count can sit still for ~40 min while work is very much happening.

    Deriving the rate from that column made the ETA read in thousands of hours
    at the start of every run.  It must track this node's own progress instead.
    """
    pbar = _GridSearchProgress(total=1000, initial=100)
    try:
        # global_done has not moved off its starting value at all, yet this node
        # has finished 10 jobs in 100 s.  8 nodes => 0.8 j/s => 900/0.8 = 18:45.
        pbar.update(global_done=100, node_done=10, elapsed_s=100.0, n_nodes=8)
        postfix = pbar._pbar.postfix
    finally:
        pbar._pbar.close()

    assert 'eta=18:45' in postfix, postfix


def test_compact_results_on_empty_dir_reports_nothing_written(tmp_path):
    data_dir = tmp_path / 'data'
    data_dir.mkdir()

    n_written, unreadable = _compact_results(data_dir)

    assert n_written is None
    assert unreadable == []
