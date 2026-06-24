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

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
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

class SyntheticDataset:
    def __init__(self, n_samples=200, n_features=10, noise=0.1, seed=42):
        rng = np.random.default_rng(seed)
        X = rng.standard_normal((n_samples, n_features))
        true_weights = rng.standard_normal((n_features, 1))
        y = X @ true_weights + noise * rng.standard_normal((n_samples, 1))
        split = int(0.8 * n_samples)
        self.X_train, self.y_train = X[:split], y[:split].ravel()
        self.X_test,  self.y_test  = X[split:], y[split:].ravel()
        self.n_features = n_features


# ---------------------------------------------------------------------------
# Params / job
# ---------------------------------------------------------------------------

class SimpleParams(BaseModel):
    out_path: Path = Path("out/simple_grid_search_results")
    alpha: float = 1e-3       # ridge regularisation
    n_samples: int = 200
    n_features: int = 10
    noise: float = 0.1
    seed: int = 42


class SimpleRidgeJob(JobInterface):
    def __init__(self, i, j, total_configs, total_samples, locks, params: SimpleParams):
        super().__init__(i, j, total_configs, total_samples, locks)
        self.params = params

    def _run(self, device=None):
        t0 = time()

        with self.locks[DATASET_LOCK_KEY]:
            dataset = SyntheticDataset(
                n_samples=self.params.n_samples,
                n_features=self.params.n_features,
                noise=self.params.noise,
                seed=self.params.seed,
            )

        model = Ridge(alpha=self.params.alpha)
        model.fit(dataset.X_train, dataset.y_train)
        mse = mean_squared_error(dataset.y_test, model.predict(dataset.X_test))

        elapsed = time() - t0
        logger.info(f"{self.get_log_prefix()} done in {elapsed:.2f}s, mse={mse:.4f}")
        return {
            'status': 'completed',
            'history': {
                'config':  self.i,
                'sample':  self.j,
                'alpha':   self.params.alpha,
                'mse':     mse,
                'elapsed': elapsed,
            },
        }


# ---------------------------------------------------------------------------
# Factory / callbacks
# ---------------------------------------------------------------------------

def create_param_combinations(n_alpha=2, n_features=2):
    combos = []
    for alpha in [1e-4, 1e-3, 1e-2, 1e-1][:n_alpha]:
        for n_feat in [10, 20][:n_features]:
            combos.append(SimpleParams(alpha=alpha, n_features=n_feat))
    return combos


class SimpleJobFactory:
    def __init__(self, param_combinations):
        self.param_combinations = param_combinations

    def __call__(self, i, j, total_configs, total_samples, locks):
        params = deepcopy(self.param_combinations[i])
        params.seed = params.seed + i * 1000 + j
        return SimpleRidgeJob(i=i, j=j, total_configs=total_configs,
                              total_samples=total_samples, locks=locks, params=params)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def simple_parallel_grid_search(
    output_path,
    samples_per_config: int = 2,
    n_alpha: int = 2,
    n_features: int = 2,
    cpu_memory_per_job_gb: float = 0.5,
    cpu_cores_per_job: int = 1,
):
    output_path = Path(output_path)
    param_combinations = create_param_combinations(n_alpha, n_features)

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
            n_alpha=n_configs // 2,
            n_features=2,
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
    # Test 1: single node                                                  #
    # ------------------------------------------------------------------ #
    print("\n[Test 1] Single node — completion and fast-exit on re-run")
    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir) / 'run'

        simple_parallel_grid_search(out, samples_per_config=SAMPLES,
                                    n_alpha=N_CONFIGS // 2, n_features=2)

        assert (out / RUN.completed_flag).exists(), f"{RUN.completed_flag} not written"
        results = pq.read_table(out / RUN.data_dir / RUN.compacted_file).to_pandas()
        assert len(results) == TOTAL, f"Expected {TOTAL} rows, got {len(results)}"
        print(f"  ✓ {TOTAL} jobs done, data.parquet has {len(results)} rows")

        t0 = time()
        simple_parallel_grid_search(out, samples_per_config=SAMPLES,
                                    n_alpha=N_CONFIGS // 2, n_features=2)
        elapsed = time() - t0
        assert elapsed < 5.0, f"Re-run should exit immediately, took {elapsed:.1f}s"
        print(f"  ✓ Re-run exited in {elapsed:.2f}s")

    # ------------------------------------------------------------------ #
    # Test 2: concurrent two-node                                          #
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

        assert (out / RUN.completed_flag).exists()
        df = pq.read_table(out / RUN.data_dir / RUN.compacted_file).to_pandas()
        assert len(df) == TOTAL, f"Expected {TOTAL} rows, got {len(df)}"
        dupes = df.duplicated(subset=['i', 'j']).sum()
        assert dupes == 0, f"{dupes} duplicate (i,j) rows"
        print(f"  ✓ {TOTAL} jobs done, {len(df)} rows, 0 duplicates")

    # ------------------------------------------------------------------ #
    # Test 3: resume from partial run                                      #
    # ------------------------------------------------------------------ #
    print("\n[Test 3] Resume from partial run (data/ pre-populated)")
    with tempfile.TemporaryDirectory() as tmpdir:
        import yaml

        out = Path(tmpdir) / 'run'
        out.mkdir()
        data_dir = out / 'data'
        data_dir.mkdir()

        param_combinations = create_param_combinations(N_CONFIGS // 2, 2)
        cfg = {
            'total_configs': len(param_combinations),
            'samples_per_config': SAMPLES,
            'combinations': [p.model_dump(mode='json') for p in param_combinations],
        }
        with (out / 'parameters.yaml').open('w') as f:
            yaml.dump(cfg, f, sort_keys=False)

        pre_done = TOTAL // 2
        pre_done_pairs = [(i, j) for i in range(N_CONFIGS) for j in range(SAMPLES)][:pre_done]
        fake_tbl = pa.table({
            'i':           pa.array([p[0] for p in pre_done_pairs], type=pa.int32()),
            'j':           pa.array([p[1] for p in pre_done_pairs], type=pa.int32()),
            'params_json': pa.array(['{}'] * pre_done),
        })
        pq.write_table(fake_tbl, data_dir / 'pre_done.parquet')

        simple_parallel_grid_search(out, samples_per_config=SAMPLES,
                                    n_alpha=N_CONFIGS // 2, n_features=2)

        assert (out / RUN.completed_flag).exists()
        df = pq.read_table(out / RUN.data_dir / RUN.compacted_file).to_pandas()
        assert len(df) == TOTAL, f"Expected {TOTAL} rows after compaction, got {len(df)}"
        print(f"  ✓ Ran {TOTAL - pre_done} remaining jobs, skipped {pre_done} pre-done; "
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
