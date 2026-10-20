# parallel_grid_search

Generic parallel grid search over a (config × sample) space. CPU+GPU, multi-node, resumable after crashes.

## How it works

Jobs are identified by `(i, j)` — config index × sample index. Each node claims jobs from a shared SQLite queue (`state.db`), runs them, and appends results as Parquet batch files. When all jobs complete, exactly one node merges the batch files into `data/data.parquet` and writes `GRID_SEARCH_COMPLETED.flag`.

**Run directory layout:**

| File | Purpose |
|---|---|
| `data/state.db` | Shared job queue — rebuilt from `data/*.parquet` if missing |
| `data/*.parquet` | Batch results written during the run |
| `data/data.parquet` | Compacted output written at completion |
| `GRID_SEARCH_COMPLETED.flag` | Presence means the run is done |
| `parameters.yaml` | Config snapshot — mismatch on resume aborts the run |

## Crash recovery

On restart, if `state.db` is missing it is atomically rebuilt by scanning `data/*.parquet` for completed `(i, j)` pairs. A file-based lock (`state.building.lock`) prevents duplicate reconstruction when multiple nodes restart simultaneously. The node that wins writes `state.db` and removes the lock; the others wait and then open the finished file.

A node that dies mid-run leaves its jobs `claimed`. Every node writes a heartbeat to `state.db` once a minute, and every node watches the heartbeats of the other nodes holding claims. A holder whose heartbeat has not changed for `HEARTBEAT_TIMEOUT_S` (5 min, timed on the watcher's own clock) is dead, and its claims go back to `pending`. Liveness never compares timestamps across machines, so a node with a skewed clock cannot look dead. A node that exits cleanly deletes its heartbeat row, so anything it still holds is reclaimed at the next check, with no 5-min wait.

To recover from a crash, either:
- **Re-run the grid search.** If other nodes are still running, they take over the dead node's jobs. If none are, the new node watches the dead node's heartbeat for 5 min, then reclaims its jobs, finishes them and compacts.
- **Run `reset_grid` first** (see below), then re-run. `state.db` is rebuilt from the batch Parquets, so unflushed jobs are pending at once, with no 5-min wait. Only do this when no node is running on that grid.

## Multi-node

All nodes share `output_path` on NFS/Lustre. `state.db` uses DELETE journal mode (not WAL — WAL's shared-memory file is unreliable on network filesystems). The compaction step is protected by a SQLite `BEGIN IMMEDIATE` mutex so exactly one node writes `data/data.parquet`.

**End of a run — last one out compacts.** Each node queues only about `ResourceLimits.prefetch_s` (30 s) of work ahead of its workers, so no node holds a long backlog at the end. Once nothing is left to claim, a node flushes each result as it finishes. When it has no jobs in flight it exits, as long as every other claim holder has a changing heartbeat. If one is dead, the node reclaims that node's jobs and keeps working. On exit, the node compacts if no jobs are pending or claimed; otherwise it leaves compaction to the nodes still running. If those nodes then die too, re-run the grid search to finish and compact.

## Usage

Implement `JobInterface` and wire into `generic_parallel_grid_search`:

```python
generic_parallel_grid_search(
    job_factory=MyJobFactory(...),
    total_configs=N,
    samples_per_config=S,
    output_path='out/my_run',
    save_config=lambda out: ...,                        # writes parameters.yaml
    process_results=lambda entries, marks, path: ...,   # writes one batch Parquet
)
```

## Resetting a run

```bash
. boolean_reservoir/cluster/cluster_utils.sh
reset_grid /out/my_run
```

Deletes `state.db`, `GRID_SEARCH_COMPLETED.flag`, and any stale lock/tmp files. Batch Parquets in `data/` are preserved — a re-run rebuilds state from them and goes straight to compaction without re-computing any jobs. Batch files are only deleted once `data.parquet` is confirmed complete (row count matches total jobs).
