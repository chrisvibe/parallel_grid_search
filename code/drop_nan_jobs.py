"""Drop the jobs whose logged result holds a NaN, so a restart re-runs them.

A job's params_json is written by json.dumps, which writes NaN for a failed fit; strict JSON
(orjson, the loader) rejects it, so such a row stops every analysis. This rewrites each batch
Parquet without those rows and keeps the original next to it as <name>.parquet.nanbak (a suffix
the state rebuild and the loader do not glob). Then run reset_grid.py on the run directory: the
job queue is rebuilt from the remaining Parquets, and the dropped jobs are queued again.

Dry run by default (lists the rows); --apply rewrites.

Usage:
    python drop_nan_jobs.py <data_dir> [--apply]
"""
import argparse
from pathlib import Path

import orjson
import pyarrow as pa
import pyarrow.parquet as pq


def nan_rows(table: pa.Table) -> list[int]:
    """Row positions whose params_json strict JSON rejects."""
    bad = []
    for n, js in enumerate(table.column('params_json').to_pylist()):
        try:
            orjson.loads(js)
        except orjson.JSONDecodeError:
            bad.append(n)
    return bad


def drop_nan_jobs(data_dir: Path, apply: bool) -> list[tuple]:
    dropped = []
    for f in sorted(data_dir.glob('*.parquet')):
        table = pq.read_table(f)
        bad = nan_rows(table)
        if not bad:
            continue
        ij = [(f.name, int(table.column('i')[n].as_py()), int(table.column('j')[n].as_py())) for n in bad]
        dropped += ij
        print(f"  {f.name}: {len(bad)} NaN row(s): {[(i, j) for _, i, j in ij]}")
        if apply:
            keep = sorted(set(range(table.num_rows)) - set(bad))
            tmp = f.with_name(f.name + '.tmp')
            pq.write_table(table.take(pa.array(keep)), tmp)
            f.rename(f.with_name(f.name + '.nanbak'))
            tmp.rename(f)
    return dropped


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('data_dir', type=Path)
    ap.add_argument('--apply', action='store_true', help='rewrite the Parquets (default: dry run)')
    args = ap.parse_args()
    rows = drop_nan_jobs(args.data_dir, args.apply)
    print(f"{len(rows)} NaN job(s) {'dropped' if args.apply else 'found (dry run; --apply to drop)'}")
