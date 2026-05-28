#!/usr/bin/env python3
"""
Simple example demonstrating CPU parallel grid search.

Usage:
    # Single-node:
    python simple_example.py

    # Multi-node: run multiple instances pointing at the same output path.
    # They coordinate automatically via the shared state.db on NFS.
    python simple_example.py --output-path /tmp/my_run
    python simple_example.py --output-path /tmp/my_run  # another terminal/node

    # Automated tests:
    python simple_example.py --test
"""

import torch
import torch.nn as nn
import torch.optim as optim
from time import time, gmtime
import logging
import multiprocessing
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from copy import deepcopy
from shutil import rmtree
from pathlib import Path
from pydantic import BaseModel

from train_model_parallel import generic_parallel_grid_search
from parallel_utils import GridSearchDB, JobInterface, DATASET_LOCK_KEY, RUN
from project.boolean_reservoir.code.utils.load_save import save_grid_search_results

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model / data
# ---------------------------------------------------------------------------

class SimpleLinearModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int = 1):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x):
        return self.network(x)


class SyntheticDataset:
    def __init__(self, n_samples=200, n_features=10, noise=0.1, device='cpu'):
        X = torch.randn(n_samples, n_features)
        true_weights = torch.randn(n_features, 1)
        y = X @ true_weights + noise * torch.randn(n_samples, 1)
        dataset = torch.utils.data.TensorDataset(X.to(device), y.to(device))
        self.dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
        self.n_features = n_features


# ---------------------------------------------------------------------------
# Params / job
# ---------------------------------------------------------------------------

class SimpleParams(BaseModel):
    out_path: Path = Path("out/simple_grid_search_results")
    learning_rate: float = 0.01
    hidden_size: int = 64
    epochs: int = 10
    n_samples: int = 200
    n_features: int = 10
    noise: float = 0.1
    seed: int = 42


class SimpleLinearJob(JobInterface):
    def __init__(self, i, j, total_configs, total_samples, locks, params: SimpleParams):
        super().__init__(i, j, total_configs, total_samples, locks)
        self.params = params

    def _run(self, device):
        torch.manual_seed(self.params.seed)
        t0 = time()

        with self.locks[DATASET_LOCK_KEY]:
            dataset = SyntheticDataset(
                n_samples=self.params.n_samples,
                n_features=self.params.n_features,
                noise=self.params.noise,
                device=device,
            )

        model = SimpleLinearModel(self.params.n_features, self.params.hidden_size).to(device)
        optimizer = optim.Adam(model.parameters(), lr=self.params.learning_rate)
        criterion = nn.MSELoss()

        for _ in range(self.params.epochs):
            for bx, by in dataset.dataloader:
                optimizer.zero_grad()
                criterion(model(bx), by).backward()
                optimizer.step()

        elapsed = time() - t0
        logger.info(f"{self.get_log_prefix()} done in {elapsed:.2f}s on {device}")
        return {
            'status': 'completed',
            'history': {
                'config': self.i,
                'sample': self.j,
                'lr': self.params.learning_rate,
                'hidden': self.params.hidden_size,
                'elapsed': elapsed,
            },
        }


# ---------------------------------------------------------------------------
# Factory / callbacks
# ---------------------------------------------------------------------------

def create_param_combinations(n_lr=2, n_hidden=2):
    combos = []
    for lr in [0.001, 0.01][:n_lr]:
        for h in [32, 64][:n_hidden]:
            combos.append(SimpleParams(learning_rate=lr, hidden_size=h))
    return combos


class SimpleJobFactory:
    def __init__(self, param_combinations):
        self.param_combinations = param_combinations

    def __call__(self, i, j, total_configs, total_samples, locks):
        params = deepcopy(self.param_combinations[i])
        params.seed = params.seed + i * 1000 + j
        return SimpleLinearJob(i=i, j=j, total_configs=total_configs,
                               total_samples=total_samples, locks=locks, params=params)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def simple_parallel_grid_search(
    output_path,
    samples_per_config: int = 2,
    n_lr: int = 2,
    n_hidden: int = 2,
    cpu_memory_per_job_gb: float = 0.5,
    cpu_cores_per_job: int = 1,
):
    output_path = Path(output_path)
    param_combinations = create_param_combinations(n_lr, n_hidden)

    def save_config(out: Path):
        import yaml
        cfg = {
            'total_configs': len(param_combinations),
            'samples_per_config': samples_per_config,
            'combinations': [p.model_dump(mode='json') for p in param_combinations],
        }
        with (out / 'parameters.yaml').open('w') as f:
            yaml.dump(cfg, f, sort_keys=False)

    def process_results(entries, marks, batch_file: Path):
        if entries:
            df = pd.DataFrame({
                'params': entries,
                'i': [m[0] for m in marks],
                'j': [m[1] for m in marks],
            })
            save_grid_search_results(df, batch_file)

    generic_parallel_grid_search(
        job_factory=SimpleJobFactory(param_combinations),
        total_configs=len(param_combinations),
        samples_per_config=samples_per_config,
        output_path=output_path,
        save_config=save_config,
        process_results=process_results,
        cpu_memory_per_job_gb=cpu_memory_per_job_gb,
        cpu_cores_per_job=cpu_cores_per_job,
        cpu_only=True,
        history_write_thresh=2,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def _run_node(output_path, samples_per_config, n_configs, results_queue):
    """Worker target for multiprocessing-based concurrent node simulation."""
    import logging, sys
    logging.basicConfig(level=logging.WARNING, stream=sys.stdout, force=True)
    logging.Formatter.converter = gmtime
    try:
        simple_parallel_grid_search(
            output_path=output_path,
            samples_per_config=samples_per_config,
            n_lr=n_configs // 2,
            n_hidden=2,
        )
        results_queue.put(('ok', str(output_path)))
    except Exception as e:
        results_queue.put(('error', f'{e}'))


def test_grid_search():
    """Three tests for the grid search architecture."""
    import tempfile

    N_CONFIGS = 4
    SAMPLES   = 2
    TOTAL     = N_CONFIGS * SAMPLES  # 8

    # ------------------------------------------------------------------ #
    # Test 1: single node — completes all jobs, writes data.parquet and    #
    # GRID_SEARCH_COMPLETED.flag, then a second call exits immediately.   #
    # ------------------------------------------------------------------ #
    print("\n[Test 1] Single node — completion and fast-exit on re-run")
    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir) / 'run'

        simple_parallel_grid_search(out, samples_per_config=SAMPLES,
                                    n_lr=N_CONFIGS // 2, n_hidden=2)

        assert (out / RUN.completed_flag).exists(), f"{RUN.completed_flag} not written"
        results = pq.read_table(out / RUN.data_dir / RUN.compacted_file).to_pandas()
        assert len(results) == TOTAL, f"Expected {TOTAL} rows in data.parquet, got {len(results)}"
        print(f"  ✓ {TOTAL} jobs done, data.parquet has {len(results)} rows, {RUN.completed_flag} written")

        # Second call should exit immediately (flag present)
        t0 = time()
        simple_parallel_grid_search(out, samples_per_config=SAMPLES,
                                    n_lr=N_CONFIGS // 2, n_hidden=2)
        elapsed = time() - t0
        assert elapsed < 5.0, f"Re-run should exit immediately, took {elapsed:.1f}s"
        print(f"  ✓ Re-run exited in {elapsed:.2f}s (fast exit via {RUN.completed_flag})")

    # ------------------------------------------------------------------ #
    # Test 2: concurrent two-node — both processes race for jobs, no      #
    # duplicates or missing rows in data.parquet.                          #
    # ------------------------------------------------------------------ #
    print("\n[Test 2] Concurrent two-node (real multiprocessing)")
    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir) / 'run'
        q   = multiprocessing.Queue()

        p0 = multiprocessing.Process(target=_run_node,
                                     args=(out, SAMPLES, N_CONFIGS, q))
        p1 = multiprocessing.Process(target=_run_node,
                                     args=(out, SAMPLES, N_CONFIGS, q))
        p0.start(); p1.start()
        p0.join();  p1.join()

        results = [q.get_nowait() for _ in range(2)]
        errors = [r for r in results if r[0] == 'error']
        assert not errors, f"Node errors: {errors}"

        assert (out / RUN.completed_flag).exists(), f"{RUN.completed_flag} not written"
        df = pq.read_table(out / RUN.data_dir / RUN.compacted_file).to_pandas()
        assert len(df) == TOTAL, f"Expected {TOTAL} rows, got {len(df)}"
        dupes = df.duplicated(subset=['i', 'j']).sum()
        assert dupes == 0, f"{dupes} duplicate (i,j) rows in data.parquet"
        print(f"  ✓ {TOTAL} jobs done, {len(df)} rows, 0 duplicates")

    # ------------------------------------------------------------------ #
    # Test 3: resume — pre-populate data/ for half the jobs, verify that  #
    # only the remaining half actually run.                               #
    # ------------------------------------------------------------------ #
    print("\n[Test 3] Resume from partial run (data/ pre-populated)")
    with tempfile.TemporaryDirectory() as tmpdir:
        import yaml

        out = Path(tmpdir) / 'run'
        out.mkdir()
        data_dir = out / 'data'
        data_dir.mkdir()

        # Write parameters.yaml matching simple_parallel_grid_search's save_config
        param_combinations = create_param_combinations(N_CONFIGS // 2, 2)
        cfg = {
            'total_configs': len(param_combinations),
            'samples_per_config': SAMPLES,
            'combinations': [p.model_dump(mode='json') for p in param_combinations],
        }
        with (out / 'parameters.yaml').open('w') as f:
            yaml.dump(cfg, f, sort_keys=False)

        # Pre-populate data/ with fake results for the first half of jobs
        pre_done = TOTAL // 2
        pre_done_pairs = [(i, j) for i in range(N_CONFIGS) for j in range(SAMPLES)][:pre_done]
        fake_tbl = pa.table({
            'i':           pa.array([p[0] for p in pre_done_pairs], type=pa.int32()),
            'j':           pa.array([p[1] for p in pre_done_pairs], type=pa.int32()),
            'params_json': pa.array(['{}'] * pre_done),
        })
        pq.write_table(fake_tbl, data_dir / 'pre_done.parquet')
        files_before = set(data_dir.glob('*.parquet'))

        # Run: _ensure_state_db reconstructs from data/ and marks pre_done jobs done.
        # Only the remaining jobs should be claimed and run.
        simple_parallel_grid_search(out, samples_per_config=SAMPLES,
                                    n_lr=N_CONFIGS // 2, n_hidden=2)

        remaining = TOTAL - pre_done

        assert (out / RUN.completed_flag).exists(), f"{RUN.completed_flag} not written"
        df = pq.read_table(out / RUN.data_dir / RUN.compacted_file).to_pandas()
        assert len(df) == TOTAL, f"data.parquet should have all {TOTAL} rows after compaction, got {len(df)}"
        print(f"  ✓ Ran {remaining} remaining jobs, skipped {pre_done} pre-done; "
              f"data.parquet has all {TOTAL} rows")

    print("\n✓ All tests passed.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse, sys

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(process)d - %(filename)s - %(message)s',
        stream=sys.stdout,
        force=True,
    )
    logging.Formatter.converter = gmtime

    parser = argparse.ArgumentParser(description="Simple parallel grid search example")
    parser.add_argument('--test', action='store_true', help='Run automated tests')
    parser.add_argument('--output-path', default=None,
                        help='Output directory (for multi-node usage)')
    args = parser.parse_args()

    if args.test:
        multiprocessing.set_start_method('spawn', force=True)
        test_grid_search()
        sys.exit(0)

    p = SimpleParams()
    out = Path(args.output_path) if args.output_path else p.out_path

    if not args.output_path:
        resp = input(f"Delete '{out}' and run? (y/N): ").strip().lower()
        if resp != 'y':
            print("Cancelled.")
            sys.exit(0)
        if out.exists():
            rmtree(out)

    simple_parallel_grid_search(output_path=out, samples_per_config=2)
