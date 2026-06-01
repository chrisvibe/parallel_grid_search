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

## Multi-node

All nodes share `output_path` on NFS/Lustre. `state.db` uses DELETE journal mode (not WAL — WAL's shared-memory file is unreliable on network filesystems). The compaction step is protected by a SQLite `BEGIN IMMEDIATE` mutex so exactly one node writes `data/data.parquet`.

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
