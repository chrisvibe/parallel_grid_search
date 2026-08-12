"""Tail-of-run correctness for the multi-node grid search.

These cover the failure that let a grid finish "complete" with jobs unwritten:
nodes retired the moment their local queue drained, stranding claims nobody
picked up, and compaction then wrote the completion flag over the resulting gap
and deleted the batch files that proved it.

Everything here is DB- and file-level, so it runs in a second without a cluster,
a dataset, or a training step.
"""
import sqlite3

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from project.parallel_grid_search.code.parallel_utils import (
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


def test_count_live_nodes_ignores_the_compaction_sentinel(db):
    """The sentinel shares the heartbeats table but is not a node doing work;
    counting it would inflate the ETA's assumed parallelism."""
    assert db.count_live_nodes() == 1, "never reports zero — this node is alive"

    db.update_heartbeat('nodeA')
    db.update_heartbeat('nodeB')
    assert db.count_live_nodes() == 2

    assert db.try_claim_compaction()
    assert db.count_live_nodes() == 2, "sentinel must not be counted as a node"


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
