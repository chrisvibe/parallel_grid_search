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

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from project.parallel_grid_search.code.parallel_utils import (
    COMPACT_SENTINEL,
    GridSearchDB,
    JobGenerator,
    JobInterface,
    STATUS_CLAIMED,
    STATUS_DONE,
    STATUS_FAILED,
    STATUS_PENDING,
)
from project.parallel_grid_search.code.train_model_parallel import (
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
    db.reset_stale_claimed(timeout_s=0)
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
# Staleness compares one node's timestamp against another's, so the cluster's
# clocks are assumed synchronised (SLURM requires this too).  A node that drifts
# further than the timeout reclaims its peers' in-flight jobs — duplicating work
# rather than losing it, since rows are unique on (i,j) — so the assumption is
# checked at startup and reported rather than silently trusted.
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


def test_a_peer_that_stops_beating_has_its_jobs_reclaimed(tmp_path):
    """A crashed node's claims must come back, or the grid can never finish them."""
    observer, peer = _two_nodes(tmp_path)
    _beat(observer, 'peer', time.time() - 400)   # last beat well past the 300 s timeout

    assert observer.reset_stale_claimed() == 3
    assert observer.counts().get('pending') == 3
    assert observer.counts().get('claimed') == 3, "a beating node keeps its work"
    peer.close()


def test_a_clean_exit_hands_its_claims_back_at_once(tmp_path):
    """close() drops the heartbeat row, so a graceful shutdown needs no timeout wait."""
    observer, peer = _two_nodes(tmp_path)
    peer.close()

    assert observer.reset_stale_claimed() == 3, "no row, no wait"


def test_the_sweep_drops_the_rows_it_declares_dead(tmp_path):
    """Otherwise every crashed node leaves a row behind for good and the table grows
    by one row per node per run."""
    observer, peer = _two_nodes(tmp_path)
    _beat(observer, 'peer', time.time() - 400)

    observer.reset_stale_claimed()

    assert 'peer' not in observer._read_heartbeats()
    assert 'observer' in observer._read_heartbeats(), "a live node's row survives"
    peer.close()


def test_a_skewed_node_stops_reclaiming_its_peers_work(tmp_path):
    """The hpc9 failure: a node 78 min fast reads every healthy peer as long dead.

    It must keep working and keep flushing — it just may not judge anyone.
    """
    observer, peer = _two_nodes(tmp_path)
    skew = 78 * 60
    _beat(observer, 'peer', time.time() - skew)
    observer._measure_clock_offset()                       # baseline
    _beat(observer, 'peer', time.time() - skew + 60)       # peer beats again: it is alive
    observer._measure_clock_offset()                       # advance proves the skew

    assert observer.is_clock_skewed()
    assert observer.reset_stale_claimed() == 0, "a skewed node must not reclaim"
    assert observer.counts().get('claimed') == 6, "the peer keeps its work"
    peer.close()


def test_one_bad_clock_cannot_drag_the_cluster_onto_its_own_time(tmp_path):
    """Observed in production: with the newest beat as reference, the fastest clock
    becomes the time authority.  One node 78 min ahead made all eight healthy nodes
    measure themselves as behind, correct into the future, and bench themselves —
    leaving nobody to reclaim crashed work.  The median lets the majority win.
    """
    healthy = GridSearchDB.open(tmp_path / 'state.db', 2, 3, node_id='healthy')
    for n in range(7):
        _beat(healthy, f'peer{n}', time.time() - 20)
    _beat(healthy, 'skewed', time.time() + 78 * 60)      # the one bad clock
    healthy._measure_clock_offset()
    for n in range(7):                                   # the crew beats again
        _beat(healthy, f'peer{n}', time.time())
    _beat(healthy, 'skewed', time.time() + 78 * 60 + 60)
    healthy._measure_clock_offset()

    assert not healthy.is_clock_skewed(), "the majority is the reference, not the outlier"
    assert healthy.reset_stale_claimed() == 0, "and it still sweeps normally"


def test_a_dead_cluster_is_not_mistaken_for_skew(tmp_path):
    """The trap in this design: an old peer timestamp reads identically whether our
    clock is fast or that peer died.  Guessing 'skew' would disable crash recovery,
    so an offset only counts once a beat has been watched to advance."""
    observer, peer = _two_nodes(tmp_path)
    _beat(observer, 'peer', time.time() - 4000)   # crashed long ago; never beats again
    observer._measure_clock_offset()
    observer._measure_clock_offset()              # look twice: nothing advances

    assert not observer.is_clock_skewed(), "no advance, no proof — assume our clock is fine"
    assert observer.reset_stale_claimed() == 3, "so the dead node's work still comes back"
    peer.close()


def test_a_skewed_node_writes_its_heartbeat_in_cluster_time(tmp_path):
    """Otherwise a fast clock stamps its rows in the future and its own jobs stay
    locked after a crash for as long as the skew lasts."""
    observer, peer = _two_nodes(tmp_path)
    skew = 78 * 60
    _beat(observer, 'peer', time.time() - skew)
    observer._measure_clock_offset()
    _beat(observer, 'peer', time.time() - skew + 60)
    observer._measure_clock_offset()

    observer.update_heartbeat('observer')
    written = observer._read_heartbeats()['observer']

    assert abs(written - (time.time() - skew)) < 90, "written in the peers' time frame"
    peer.close()


def test_an_unskewed_node_writes_its_own_clock_untouched(tmp_path):
    """Healthy clusters must behave exactly as before — no correction, no drift."""
    observer, peer = _two_nodes(tmp_path)
    _beat(observer, 'peer', time.time() - 5)
    observer._measure_clock_offset()
    _beat(observer, 'peer', time.time())
    observer._measure_clock_offset()

    assert not observer.is_clock_skewed()
    observer.update_heartbeat('observer')

    assert abs(observer._read_heartbeats()['observer'] - time.time()) < 5
    peer.close()


def test_clock_skew_is_reported_but_nothing_depends_on_it(tmp_path):
    """The startup warning is the whole defence against the synchronised-clock
    assumption being violated silently."""
    observer = GridSearchDB.open(tmp_path / 'state.db', 2, 3, node_id='observer')
    assert observer.clock_skew_s() is None, "nobody else has beaten yet"

    _beat(observer, 'peer', time.time() - 78 * 60)
    skew = observer.clock_skew_s()

    assert skew is not None and 78 * 60 - 5 < skew < 78 * 60 + 5
    observer.close()


def test_the_reported_skew_matches_the_one_decisions_use(tmp_path):
    """They must share a reference.  While the warning measured against the newest
    beat and the sweep against the median, one skewed peer made every healthy node
    log itself as 77 min behind — the opposite of the truth."""
    healthy = GridSearchDB.open(tmp_path / 'state.db', 2, 3, node_id='healthy')
    for n in range(7):
        _beat(healthy, f'peer{n}', time.time())
    _beat(healthy, 'skewed', time.time() + 78 * 60)

    assert abs(healthy.clock_skew_s()) < 5, "the outlier must not set the reference"
    healthy.close()


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
