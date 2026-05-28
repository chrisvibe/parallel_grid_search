# parallel_grid_search

Generic parallel grid search over (config × sample) grids. CPU+GPU, multi-node, resumable.

## Run directory

| File | Role |
|---|---|
| `log.db` | Job state — `pending / claimed / done` for every `(i, j)` |
| `data.parquet` | Results — one row per completed job with columns `i`, `j`, `params_json` |
| `parquet.lock.db` | Cross-node mutex for Parquet writes |
| `parameters.yaml` | Config snapshot — mismatch on resume aborts the run |

## Data integrity guarantees

| Concern | Mechanism |
|---|---|
| Concurrent writes from multiple nodes | `parquet.lock.db` (SQLite `BEGIN IMMEDIATE`) |
| Corrupt parquet from mid-write kill | Atomic write: temp file → `os.replace()` |
| Corrupt `log.db` from mid-write kill | Deleted; W derived from parquet `(i,j)` columns; parquet trimmed to W; `log.db` rebuilt |
| Duplicate results after recovery | Parquet trimmed to W before re-running above-watermark jobs |
| Duplicate results from retried jobs | `load_params_df` deduplicates by `(i, j)` on every read |

Writes are sequenced so `log.db` and `watermark.db` are never mid-write simultaneously —
a single kill corrupts at most one of them, and the intact file drives recovery.

## Usage

Implement `JobInterface._run()` and wire into `generic_parallel_grid_search`:

```python
generic_parallel_grid_search(
    job_factory=MyJobFactory(...),
    total_configs=N,
    samples_per_config=S,
    output_path='out/my_run',
    save_config=lambda out: ...,             # writes parameters.yaml
    process_results=lambda h, m, out, done: ...,  # h=history, m=marks (i,j pairs)
    recover_results=lambda out, S: [...],    # optional: derive W from parquet, trim, return done pairs
)
```

See `code/simple_example.py` for a runnable end-to-end example.

## Multi-node (SLURM)

Each node runs the same module; `TOTAL_NODES` and `SLURMD_NODENAME` are set by SLURM.
`run_on_nodes(configs, run_fn)` reads these env vars and passes `node_id` to the grid search.
All nodes share `output_path` on NFS/Lustre. `log.db` uses DELETE journal mode (not WAL)
because WAL's shared-memory file is unreliable on network filesystems.
