#!/usr/bin/env python3
"""Recursively find and check all grid search outputs. Usage: python check_grid.py <root_dir>"""
import sqlite3
import sys
from pathlib import Path
import pyarrow.parquet as pq

_STATUS_NAMES = {0: 'pending', 1: 'claimed', 2: 'done'}

SYM_OK   = '✓'
SYM_WARN = '!'
SYM_BAD  = '✗'
SYM_NONE = ' '


def line(sym, label, text):
    print(f'  {sym} {label:>7}: {text}')


def check_one(d: Path):
    data_dir = d / 'data'
    db_path = data_dir / 'state.db'

    with sqlite3.connect(db_path) as conn:
        raw = conn.execute(
            'SELECT status, COUNT(*) FROM jobs GROUP BY status ORDER BY COUNT(*) DESC'
        ).fetchall()
    rows = [(_STATUS_NAMES.get(s, str(s)), n) for s, n in raw]
    total = sum(n for _, n in rows)
    done_db = next((n for s, n in rows if s == 'done'), 0)
    complete = done_db == total

    order = {'done': 0, 'claimed': 1, 'pending': 2}
    for status, n in sorted(rows, key=lambda r: order.get(r[0], 99)):
        sym = SYM_OK if (status == 'done' and complete) else SYM_NONE
        line(sym, status, f'{n}/{total}')

    try:
        p = data_dir / 'data.parquet'
        tmp = data_dir / 'data.parquet.tmp'
        batch_files = sorted(f for f in data_dir.glob('*.parquet') if f.name != 'data.parquet')

        if tmp.exists():
            line(SYM_WARN, 'parquet', '.tmp exists (interrupted write?)')

        if p.exists():
            try:
                tbl = pq.read_table(p, columns=['i', 'j'])
                pairs = set(zip(tbl['i'].to_pylist(), tbl['j'].to_pylist()))
                n_log = len(pairs)
                n_total = len(tbl)
            except Exception as e:
                line(SYM_BAD, 'parquet', f'CORRUPT — {e}')
                pairs = set()
                n_log = 0
                n_total = 0

            # Merge with any unconsolidated batch files to get true coverage
            batch_n_total = 0
            for f in batch_files:
                try:
                    bt = pq.read_table(f, columns=['i', 'j'])
                    for i, j in zip(bt['i'].to_pylist(), bt['j'].to_pylist()):
                        pairs.add((i, j))
                    batch_n_total += len(bt)
                except Exception:
                    pass

            n_unique = len(pairs)
            dups = (n_total + batch_n_total) - n_unique
            # Only show breakdown when batch files add new coverage
            if batch_files and n_unique > n_log:
                line(SYM_NONE, 'parquet', f'{n_log} rows  (+{n_unique - n_log} in {len(batch_files)} batch files → {n_unique} total)')
            if n_unique < done_db:
                line(SYM_BAD, 'parquet', f'{n_unique} unique rows  DATA LOSS (db says {done_db}, missing {done_db - n_unique})')
            elif n_unique > done_db:
                line(SYM_BAD, 'parquet', f'{n_unique} unique rows  ahead by {n_unique - done_db}')
            else:
                line(SYM_OK if complete else SYM_NONE, 'parquet', f'{n_unique} unique rows')
            if dups:
                line(SYM_WARN, 'parquet', f'{dups} duplicates from restarts')
            if batch_files and n_unique == done_db:
                line(SYM_WARN, 'batches', f'{len(batch_files)} batch files not yet compacted into data.parquet')
        elif batch_files:
            pairs = set()
            n_total = 0
            for f in batch_files:
                tbl = pq.read_table(f, columns=['i', 'j'])
                for i, j in zip(tbl['i'].to_pylist(), tbl['j'].to_pylist()):
                    pairs.add((i, j))
                n_total += len(tbl)
            n_unique = len(pairs)
            dups = n_total - n_unique
            line(SYM_NONE, 'batches', f'{len(batch_files)} files, {n_unique} unique rows (not yet compacted)')
            if n_unique < done_db:
                line(SYM_WARN, 'batches', f'missing {done_db - n_unique} vs db (in-progress or lost)')
            if dups:
                line(SYM_WARN, 'batches', f'{dups} duplicates from restarts')
        else:
            sym = SYM_BAD if done_db > 0 else SYM_NONE
            msg = f'DATA LOSS ({done_db} done in DB but no parquet)' if done_db > 0 else 'not started'
            line(sym, 'parquet', msg)
    except ImportError:
        pass
    except Exception as e:
        line(SYM_BAD, 'parquet', f'ERROR — {e}')


root = Path(sys.argv[1])
# state.db always lives in data/ — find it there
dbs = sorted(root.rglob('data/state.db'))
if not dbs:
    print(f'No data/state.db found under {root}')
    sys.exit(1)

for db_path in dbs:
    d = db_path.parent.parent  # data/state.db → go up twice to grid search root
    print(d)
    check_one(d)
    print()
