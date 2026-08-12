"""Reset a grid search run directory so it can be re-run or re-compacted.

Removes state files and stale locks/tmps. Never touches readable batch
Parquets or data.parquet — those are preserved so a re-run rebuilds state
from them and goes straight to compaction without recomputing any jobs.

Parquets that pyarrow cannot read are quarantined (renamed to
<name>.corrupt) so state reconstruction recomputes their jobs instead of
crashing on them. Requires pyarrow; without it the check is skipped with
a warning. Verification reads each Parquet in full.

Files removed (must stay in sync with RunLayout in parallel_utils.py):
  state.db              — job queue; rebuilt from data/*.parquet on restart
  state.db.tmp          — stale if a node crashed mid-reconstruction
  GRID_SEARCH_COMPLETED.flag — cleared so the run can continue/recompact
  state.building.lock   — stale if a node crashed mid-reconstruction
  data.parquet.tmp      — stale if a node crashed mid-compaction

Usage:
    python reset_grid.py <output_dir> [<output_dir> ...]
"""
import sys
from pathlib import Path

# Must stay in sync with RunLayout in parallel_utils.py
_RESET_NAMES = frozenset({
    'state.db',
    'state.db.tmp',
    'GRID_SEARCH_COMPLETED.flag',
    'state.building.lock',
    'data.parquet.tmp',
})


def quarantine_corrupt_parquets(path: Path) -> int:
    """Rename unreadable Parquets under path to <name>.corrupt. Returns count."""
    try:
        import pyarrow.parquet as pq
    except ImportError:
        print("  warning: pyarrow not available — corrupt-Parquet check skipped")
        return 0
    n = 0
    for f in sorted(path.rglob('*.parquet')):
        try:
            pq.read_table(f)
        except Exception as e:
            quarantined = f.with_name(f.name + '.corrupt')
            f.rename(quarantined)
            print(f"  quarantined corrupt Parquet: {f} -> {quarantined.name} ({e})")
            n += 1
    return n


def reset_grid(path: Path) -> int:
    """Delete reset-eligible files under path. Returns number of files deleted."""
    n = 0
    for name in sorted(_RESET_NAMES):
        for f in sorted(path.rglob(name)):
            f.unlink()
            print(f"  deleted: {f}")
            n += 1
    n += quarantine_corrupt_parquets(path)
    return n


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    total = 0
    for arg in sys.argv[1:]:
        p = Path(arg)
        if not p.exists():
            print(f"warning: {p} does not exist, skipping")
            continue
        print(f"Resetting {p} ...")
        total += reset_grid(p)

    if total == 0:
        print("Nothing to reset.")
    else:
        print(f"\nDone — {total} file(s) removed.")
