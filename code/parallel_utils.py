import contextlib
import dataclasses
import random
import sqlite3
import threading
import time

import logging as _logging
_db_logger = _logging.getLogger(__name__)

def _db_retry(fn, max_retries: int = 10, base_wait: float = 0.5):
    """Call fn() and retry on sqlite3.OperationalError with exponential back-off.

    Covers transient NFS lock contention that exceeds SQLite's built-in
    busy_timeout.  Total wait budget ≈ sum(k*base_wait for k in 1..max_retries)
    ≈ 27 s with defaults — enough for 5-node concurrent parquet flushes.
    """
    for attempt in range(max_retries):
        try:
            return fn()
        except sqlite3.OperationalError:
            if attempt == max_retries - 1:
                raise
            wait = (attempt + 1) * base_wait + random.random()
            _db_logger.warning(
                f"DB locked (attempt {attempt + 1}/{max_retries}), retrying in {wait:.1f}s"
            )
            time.sleep(wait)
import psutil
from collections import defaultdict, deque
from os import environ, sched_getaffinity
from pathlib import Path
from abc import ABC, abstractmethod
from typing import Callable, Dict, Any
import numpy as np

import logging
logger = logging.getLogger(__name__)

from project.parallel_grid_search.code.run_layout import RunLayout, RUN

DATASET_LOCK_KEY = 'dataset_lock'


class SQLiteLock:
    """Advisory mutex backed by SQLite BEGIN IMMEDIATE.

    Works across all processes on this node and across nodes sharing the same
    NFS/Lustre path.

    Picklable (stores only path) — each worker process unpickles its own
    instance and opens a fresh connection on acquire(), which is what
    spawn-mode workers need.
    """

    def __init__(self, db_path: Path, timeout_ms: int = 30_000):
        self._path = str(db_path)
        self._timeout_ms = timeout_ms
        self._conn = None

    def acquire(self):
        def _try():
            conn = sqlite3.connect(self._path, timeout=self._timeout_ms / 1000,
                                   isolation_level=None)
            conn.execute(f"PRAGMA busy_timeout={self._timeout_ms}")
            try:
                conn.execute("BEGIN IMMEDIATE")
            except sqlite3.OperationalError:
                conn.close()
                raise
            self._conn = conn
        _db_retry(_try, max_retries=5, base_wait=2.0)

    def release(self):
        if self._conn is not None:
            self._conn.execute("COMMIT")
            self._conn.close()
            self._conn = None

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, exc_type, *_):
        if self._conn is not None:
            try:
                self._conn.execute("ROLLBACK" if exc_type else "COMMIT")
            finally:
                self._conn.close()
                self._conn = None
        return False

    def __getstate__(self):
        return {'_path': self._path, '_timeout_ms': self._timeout_ms}

    def __setstate__(self, state):
        self._path = state['_path']
        self._timeout_ms = state.get('_timeout_ms', 30_000)
        self._conn = None


# Job status constants — stored as integers for faster SQLite comparisons and indexing.
STATUS_PENDING = 0   # not yet started
STATUS_CLAIMED = 1   # checked out by a node; result not yet written
STATUS_DONE    = 2   # result written to parquet (or permanently failed)
_STATUS_NAMES  = {STATUS_PENDING: 'pending', STATUS_CLAIMED: 'claimed', STATUS_DONE: 'done'}

# Sentinel node_id inserted into heartbeats to claim the compaction role.
# Only one node may hold this at a time; it is deleted when compaction finishes.
COMPACT_SENTINEL = '__compact__'


class GridSearchDB:
    """SQLite-backed job tracker — single source of truth for job state across nodes.

    Job lifecycle: pending (0) → claimed (1) → done (2).
    status is an INTEGER column; a B-tree index makes pending-job lookups O(log n)
    rather than a full table scan — important for large nearly-complete grids.

    WAL mode is NOT used (the -shm shared-memory file doesn't propagate reliably over
    NFS/Lustre). DELETE journal mode uses fcntl byte-range locks, which Lustre supports.
    """

    _SCHEMA = """
        CREATE TABLE IF NOT EXISTS jobs (
            i       INTEGER NOT NULL,
            j       INTEGER NOT NULL,
            status  INTEGER NOT NULL DEFAULT 0,
            node_id TEXT,
            PRIMARY KEY (i, j)
        );
        CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
        CREATE TABLE IF NOT EXISTS heartbeats (
            node_id   TEXT NOT NULL PRIMARY KEY,
            last_seen REAL NOT NULL
        );
    """

    # How long (seconds) without a heartbeat before a node is considered dead.
    HEARTBEAT_TIMEOUT_S: int = 300   # 5 × 60-second heartbeat interval

    def __init__(self, db_path: Path, node_id: str = ''):
        self.db_path = Path(db_path)
        self.node_id = node_id
        self._lock = threading.Lock()  # serialize cross-thread access to _conn
        for attempt in range(30):
            try:
                conn = sqlite3.connect(str(db_path), timeout=60, check_same_thread=False,
                                       isolation_level=None)
                conn.execute("PRAGMA busy_timeout=30000")

                # Schema migration: detect old TEXT-status schema and delete the file so
                # _ensure_state_db falls through to full reconstruction from parquets.
                col = conn.execute(
                    "SELECT type FROM pragma_table_info('jobs') WHERE name='status'"
                ).fetchone()
                if col is not None and col[0].upper() == 'TEXT':
                    logger.info(
                        "state.db has old TEXT status schema — deleting for migration "
                        "(_reconstruct_state will rebuild from data/*.parquet)"
                    )
                    conn.close()
                    for suffix in ('', '-journal', '-wal', '-shm'):
                        Path(str(db_path) + suffix).unlink(missing_ok=True)
                    # Re-open with new schema
                    conn = sqlite3.connect(str(db_path), timeout=60, check_same_thread=False,
                                           isolation_level=None)
                    conn.execute("PRAGMA busy_timeout=30000")

                conn.executescript(self._SCHEMA)  # executescript auto-commits

                # Add node_id column to jobs if upgrading from earlier INTEGER schema.
                try:
                    conn.execute("ALTER TABLE jobs ADD COLUMN node_id TEXT")
                except sqlite3.OperationalError:
                    pass  # column already exists

                self._conn = conn
                break
            except sqlite3.DatabaseError as e:
                try:
                    conn.close()
                except Exception:
                    pass
                msg = str(e).lower()
                file_size = Path(db_path).stat().st_size if Path(db_path).exists() else 0
                # Empty/absent file = NFS creation race (other node still writing schema).
                # Non-empty file with DB error = corrupted by mid-write kill → delete.
                is_race = file_size == 0 and 'not a database' in msg
                is_corrupt = 'malformed' in msg or ('not a database' in msg and file_size > 0)
                if is_race and attempt < 29:
                    wait = 1.0 + random.random()
                    logger.warning(f"DB not ready yet (attempt {attempt+1}/30), retrying in {wait:.1f}s: {e}")
                    time.sleep(wait)
                elif is_corrupt:
                    logger.warning(
                        f"state.db corrupted (killed mid-write?) — deleting and recreating "
                        f"(_reconstruct_state will rebuild from data/*.parquet): {e}"
                    )
                    for suffix in ('', '-journal', '-wal', '-shm'):
                        Path(str(db_path) + suffix).unlink(missing_ok=True)
                else:
                    raise

    @contextlib.contextmanager
    def _tx(self):
        """Thread-safe explicit transaction on the shared _conn.

        Acquires the instance lock, issues BEGIN IMMEDIATE, yields the connection,
        then COMMITs or ROLLBACKs.  BEGIN IMMEDIATE acquires a RESERVED lock at
        transaction start, preventing other nodes from starting write transactions
        concurrently (critical for NFS where DEFERRED transactions have a race window).
        """
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                yield self._conn
                self._conn.execute("COMMIT")
            except Exception:
                self._conn.execute("ROLLBACK")
                raise

    def init_jobs(self, total_configs: int, samples_per_config: int) -> None:
        """Idempotent: insert all (i,j) pairs as pending if not already present.

        Fast-path: if the job count already matches, skip the bulk INSERT entirely.
        This prevents the second node in a multi-node run from doing 290k no-op inserts.
        """
        expected = total_configs * samples_per_config
        existing = self._conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]
        if existing >= expected:
            return
        with self._tx() as conn:
            conn.executemany(
                "INSERT OR IGNORE INTO jobs (i, j, status) VALUES (?, ?, 0)",
                [(i, j)
                 for i in range(total_configs)
                 for j in range(samples_per_config)],
            )

    def claim_next_batch(self, batch_size: int) -> list:
        """Atomically claim up to batch_size pending jobs, stamping this node's id.

        Transitions status 0→1 (pending→claimed) and records node_id so heartbeat
        monitoring can reset jobs back to pending if this node dies.
        The index on status makes the inner SELECT O(log n) even on nearly-complete grids.

        Returns a list of (i, j) tuples, possibly shorter than batch_size if few remain.
        """
        with self._tx() as conn:
            rows = conn.execute(
                "UPDATE jobs SET status=1, node_id=? "
                "WHERE rowid IN (SELECT rowid FROM jobs WHERE status=0 LIMIT ?) "
                "RETURNING i, j",
                (self.node_id, batch_size),
            ).fetchall()
        return [(r[0], r[1]) for r in rows]

    def update_heartbeat(self, node_id: str) -> None:
        """Upsert a liveness timestamp for node_id.  Called every ~60 s by a daemon thread."""
        def _do():
            with self._tx() as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO heartbeats (node_id, last_seen) "
                    "VALUES (?, strftime('%s','now'))",
                    (node_id,),
                )
        _db_retry(_do)

    def reset_stale_claimed(self, timeout_s: int | None = None) -> int:
        """Reset claimed jobs whose node has stopped heartbeating back to pending.

        A job is stale when its node_id:
          - is NULL (shouldn't happen, but defensive),
          - has no heartbeat entry (node vanished without cleanup), or
          - has a heartbeat older than timeout_s seconds.

        Returns the number of jobs reset.
        """
        if timeout_s is None:
            timeout_s = self.HEARTBEAT_TIMEOUT_S
        def _do():
            with self._tx() as conn:
                return conn.execute(
                    "UPDATE jobs SET status=0, node_id=NULL "
                    "WHERE status=1 AND ("
                    "    node_id IS NULL"
                    "    OR node_id NOT IN (SELECT node_id FROM heartbeats)"
                    "    OR node_id IN (SELECT node_id FROM heartbeats"
                    "                  WHERE last_seen < strftime('%s','now') - ?)"
                    ")",
                    (timeout_s,),
                ).rowcount
        return _db_retry(_do)

    def mark_done_batch(self, pairs: list) -> None:
        """Transition status 1→2 (claimed→done) for a batch of (i,j) pairs.

        Called after results are written to parquet, and for permanently-failed jobs.
        """
        if not pairs:
            return
        def _do():
            with self._tx() as conn:
                conn.executemany(
                    "UPDATE jobs SET status=2 WHERE i=? AND j=?",
                    [(i, j) for i, j in pairs],
                )
        _db_retry(_do)

    def counts(self) -> dict:
        """Return {status_name: count} for all statuses present in the DB.

        Integer status values are mapped to their names ('pending', 'claimed', 'done')
        so callers can use .get('done', 0) etc. regardless of the underlying storage type.
        """
        def _read():
            with self._lock:
                raw = dict(self._conn.execute(
                    "SELECT status, COUNT(*) FROM jobs GROUP BY status"
                ).fetchall())
            return {_STATUS_NAMES.get(k, str(k)): v for k, v in raw.items()}
        return _db_retry(_read)

    @classmethod
    def open(cls, path: Path, total_configs: int, samples_per_config: int,
             node_id: str = '') -> 'GridSearchDB':
        """Open (or create) the DB and initialise jobs."""
        db = cls(path, node_id=node_id)
        db.init_jobs(total_configs, samples_per_config)
        return db

    def try_claim_compaction(self) -> bool:
        """Atomically claim the compaction role via the heartbeats sentinel.

        Returns True if this node won (INSERT succeeded).  All other nodes that
        call this concurrently will find the sentinel already present and get False.
        Uses _tx() (BEGIN IMMEDIATE on state.db) — no separate SQLiteLock needed.
        """
        def _do():
            with self._tx() as conn:
                conn.execute(
                    "INSERT OR IGNORE INTO heartbeats (node_id, last_seen) "
                    "VALUES (?, strftime('%s','now'))",
                    (COMPACT_SENTINEL,),
                )
                return conn.execute("SELECT changes()").fetchone()[0] == 1
        return _db_retry(_do)

    def release_compaction_claim(self) -> None:
        """Remove the compaction sentinel so future runs can compact again."""
        def _do():
            with self._tx() as conn:
                conn.execute("DELETE FROM heartbeats WHERE node_id=?", (COMPACT_SENTINEL,))
        _db_retry(_do)

    def close(self) -> None:
        with self._lock:   # wait for any in-flight _tx() to finish before closing
            self._conn.close()


class JobInterface(ABC):
    """Simple interface that all jobs must implement"""
    
    def __init__(self, i: int, j: int, total_configs: int, total_samples: int, locks: dict):
        self.i = i
        self.j = j
        self.total_configs = total_configs
        self.total_samples = total_samples
        self.locks = locks
        self._config_pad = len(str(total_configs))
        self._sample_pad = len(str(total_samples))
    
    @abstractmethod
    def _run(self, device) -> Dict[str, Any]:
        """Run the job and return results"""
        pass
    
    def run(self, device):
        logger.debug(f"Running Job {self.i}-{self.j} on device {device}")
        try:
            return self._run(device)
        except MemoryError as e:
            logger.error(f"GPU OOM for job {self.i}-{self.j}")
            return {'status': 'error', 'error_type': 'OOM', 'error': str(e)}
        except Exception as e:
            logger.exception(f"Error in job {self.i}-{self.j}: {e}")
            return {'status': 'error', 'error_type': 'general', 'error': str(e)}
    

    def get_log_prefix(self):
        """Generate log prefix with config/sample info"""
        config_str = f"{self.i+1:0{self._config_pad}d}/{self.total_configs}"
        sample_str = f"{self.j+1:0{self._sample_pad}d}/{self.total_samples}"
        return f"Config: {config_str}, Sample: {sample_str}"
    
    def __str__(self):
        return f'{self.i}-{self.j}'


@dataclasses.dataclass
class JobGenerator:
    """Iterates over the (config, sample) grid and yields jobs via job_factory.

    When db is None: sequential iteration over all (i,j) pairs (single-node mode).
    When db is set: jobs are claimed atomically from the DB (multi-node mode).
    Claiming transitions jobs to status=1 (claimed); status=2 (done) is set after the
    result is written to parquet via mark_done_batch.
    """
    job_factory: Callable
    total_configs: int
    samples_per_config: int
    db: 'GridSearchDB | None' = None
    locks: dict = dataclasses.field(default_factory=dict)

    def __len__(self) -> int:
        return self.total_configs * self.samples_per_config

    def __iter__(self):
        if self.db is not None:
            # Batch-claim jobs to amortize NFS SQLite lock acquisitions.
            # Without batching, filling a 96-job queue costs 96 round-trips;
            # with batch_size=32 it costs 3.
            _BATCH = ResourceLimits.db_claim_batch_size
            buffer: deque = deque()
            while True:
                if not buffer:
                    try:
                        batch = self.db.claim_next_batch(_BATCH)
                    except sqlite3.OperationalError as e:
                        # DB locked on NFS — brief sleep and retry WITHOUT propagating out
                        # of the generator (a propagated exception closes the generator,
                        # causing the next next() to raise StopIteration and falsely set
                        # submission_complete=True). Keep the sleep short so the feeder
                        # thread isn't blocked long enough to miss recovery steps.
                        logger.warning(f"DB locked in JobGenerator, retrying: {e}")
                        time.sleep(0.5 + random.random() * 0.5)
                        continue
                    if not batch:
                        return
                    buffer.extend(batch)
                i, j = buffer.popleft()
                yield self.job_factory(
                    i=i, j=j,
                    total_configs=self.total_configs,
                    total_samples=self.samples_per_config,
                    locks=self.locks,
                )
        else:
            for i in range(self.total_configs):
                for j in range(self.samples_per_config):
                    yield self.job_factory(
                        i=i, j=j,
                        total_configs=self.total_configs,
                        total_samples=self.samples_per_config,
                        locks=self.locks,
                    )
    
@dataclasses.dataclass
class ResourceLimits:
    """Centralized thresholds for all resource-management decisions.

    Control hierarchy (softest → hardest):

    Mechanism        Signal                   Trigger                          Action
    ───────────────  ───────────────────────  ───────────────────────────────  ──────────────────────────────
    Budget ceiling   Estimated RSS            > mem_target_fraction            Cap target worker count
    Spawn gate       Projected RSS            > mem_target_fraction            Block this spawn only
    Elevated gate    Actual job RSS           > mem_elevated_fraction          Pause feeding + backoff (halve limit)
    Critical gate    Machine RAM %            > mem_critical_machine_fraction  Backoff (halve limit)
                     SLURM actual RSS         > mem_critical_alloc_fraction    Backoff (halve limit)

    Asymmetric scaling:
      Scale down — immediate: backoff() halves limit + sends sentinels at once
      Scale up   — slow: +1 per recovery_period_s after backoff clears;
                         +1 per grow_delay_factor × recovery_period_s for budget-ceiling growth

    Hysteresis:
      Elevated gate: enter at mem_elevated_fraction, exit at mem_elevated_exit_fraction
      GPU compute:   enter saturation at gpu_compute_hi, exit at gpu_compute_lo
    """
    # Proactive concurrency ceiling — cap workers so ESTIMATED total RSS ≤ this.
    # Uses EWMA estimates, so may briefly overshoot; elevated gate catches the excess.
    mem_target_fraction: float = 0.87

    # Elevated gate (soft emergency: pause feeding + backoff).
    # Applies to available_memory_gb (SLURM alloc if Slurm, machine RAM otherwise).
    # Defaults to mem_target_fraction + 0.03 so the spawn gate fires first (stop adding
    # workers), then if RSS keeps climbing and hits elevated, workers are shed.
    mem_elevated_fraction: float | None = None   # default: mem_target_fraction + 0.03 (see __post_init__)
    # Exit threshold is well below entry to avoid rapid oscillation near the boundary.
    # Gap of ~9pp on a 32 GB machine ≈ 2.8 GB — enough to absorb one worker's variance.
    mem_elevated_exit_fraction: float = 0.78     # exit (hysteresis lower bound)

    # Critical gates (hard emergency: backoff + shed workers immediately)
    mem_critical_machine_fraction: float = 0.92  # machine-wide (all users, any env)
    mem_critical_alloc_fraction: float = 0.99    # our job's RSS vs SLURM alloc (Slurm only)

    # CPU spawn gate — blocks new worker spawns when CPU is fully saturated.
    # Higher than the memory target: high CPU means workers are running (desired),
    # so only block near 100% where a new spawn would cause context-switch overhead.
    cpu_max_usage: float = 0.92

    # EWMA smoothing (α = weight of newest sample)
    cpu_ewma_alpha: float = 0.05          # slow: downward decay (observed < estimate — conservative)
    cpu_ewma_alpha_up: float = 0.30       # fast: upward adapt (observed > estimate — protect against OOM)
    cpu_ewma_startup_alpha: float = 0.30  # fast: first N samples converge quickly (matches GPU alpha)
    cpu_ewma_startup_n: int = 10          # switch from startup→steady alpha after this many samples
    gpu_ewma_alpha: float = 0.30  # fast: responds to rapid GPU utilisation changes

    # GPU thresholds
    gpu_memory_util_max: float = 0.85
    gpu_compute_lo: float = 0.85  # hysteresis lower bound — wider band prevents thrashing
    gpu_compute_hi: float = 0.95  # hysteresis upper bound (saturate at or above this)

    # Scaling timing
    recovery_period_s: float = 30.0  # fixed interval between each +1 concurrency step during backoff recovery
    grow_delay_factor: float = 4.0   # budget-ceiling growth X× slower than recovery (120 s/step)

    # Infrastructure
    db_claim_batch_size: int = 128
    mem_cache_ttl_s: float = 0.5
    mem_estimate_update_interval_s: float = 5.0
    min_alive_for_estimate: int = 2
    startup_rss_min_fraction: float = 0.5  # skip EWMA updates below init_estimate × this
    min_memory_per_job_gb: float = 0.5     # floor for EWMA memory estimate

    def __post_init__(self):
        if self.mem_elevated_fraction is None:
            self.mem_elevated_fraction = self.mem_target_fraction + 0.03


def grouped_bar_graph(values, width=32):
    """Compact unicode bar graph — used by CPUJobResourceManager.get_status()."""
    def bar_graph(vals):
        bars = "▁▂▃▄▅▆▇█"
        return ''.join(bars[min(int(p / 100 * (len(bars) - 1)), len(bars) - 1)] for p in vals)

    values = np.array(values)
    if len(values) <= width:
        return bar_graph(values)
    chunks = np.array_split(values, width)
    return bar_graph([c.mean() for c in chunks])


class CPUJobResourceManager:
    """CPU resource manager with SLURM support"""

    def __init__(self, memory_per_job_gb=2.0, cores_per_job=1, max_workers=None, limits=None):
        self.limits = limits or ResourceLimits()
        self.memory_per_job_gb = memory_per_job_gb
        self._initial_memory_per_job_gb = memory_per_job_gb
        self.cores_per_job = cores_per_job
        self.max_workers = max_workers

        self._detect_constraints()
        self._log_initialization()

        psutil.cpu_percent(interval=None, percpu=True)  # prime non-blocking counter
        self._usage_cache: dict = {}
        self._usage_cache_time: float = 0.0
        self._cache_ttl: float = self.limits.mem_cache_ttl_s

        # All memory-tracking state initialised here — no hasattr guards needed in methods
        self._last_mem_check_time: float = 0.0
        self._cached_job_mem_gb: float = 0.0
        self._last_total_rss_gb: float = 0.0
        self._obs_last_update: float = 0.0
        self._obs_mem_ewma: float | None = None
        self._obs_n_samples: int = 0       # counts EWMA updates; startup alpha used until >= cpu_ewma_startup_n
        self._in_elevated_state: bool = False

    @staticmethod
    def _get_affinity_based_cpu_cores():
        """Get CPU cores available to this process"""
        try:
            return sched_getaffinity(0)
        except (AttributeError, OSError):
            return None

    def _detect_constraints(self):
        """Detect SLURM or other environment constraints"""
        # Physical and logical counts from system
        self.physical_cpus = psutil.cpu_count(logical=False) or 1
        self.logical_cpus = psutil.cpu_count(logical=True) or self.physical_cpus

        # SLURM detection - ONLY use SLURM_CPUS_PER_TASK (your actual allocation)
        # SLURM_CPUS_ON_NODE is the node total, not your allocation
        slurm_val = environ.get('SLURM_CPUS_PER_TASK')
        self.slurm_cpus = int(slurm_val) if slurm_val else None
        self.is_slurm = self.slurm_cpus is not None

        # available_cpus: use SLURM allocation when set (actual allocated cores),
        # otherwise fall back to logical CPUs
        self.available_cpus = self.slurm_cpus if self.slurm_cpus is not None else self.logical_cpus

        # Memory detection
        if 'SLURM_MEM_PER_NODE' in environ:
            self.available_memory_gb = int(environ['SLURM_MEM_PER_NODE']) / 1024
        elif 'SLURM_MEM_PER_CPU' in environ:
            mem_per_cpu_mb = int(environ['SLURM_MEM_PER_CPU'])
            # Use slurm allocation or physical cores (not logical - avoids double counting)
            cpus_for_mem = self.slurm_cpus or self.physical_cpus
            self.available_memory_gb = (mem_per_cpu_mb * cpus_for_mem) / 1024
        else:
            self.available_memory_gb = psutil.virtual_memory().total / (1024**3)

    def _log_initialization(self):
        """Log initialization details"""
        logger.info(f"CPU Resource Manager initialized:")
        logger.info(f"  Environment: {'SLURM' if self.is_slurm else 'Standard'}")
        if self.slurm_cpus:
            logger.info(f"  SLURM allocation: {self.slurm_cpus} CPUs")
        logger.info(f"  Physical cores: {self.physical_cpus}")
        logger.info(f"  Logical cores: {self.logical_cpus}")
        logger.info(f"  Available Memory: {self.available_memory_gb:.1f}GB")
        logger.info(f"  Memory per job: {self.memory_per_job_gb}GB")
        logger.info(f"  Cores per job: {self.cores_per_job}")
        if self.max_workers is not None:
            logger.info(f"  Max workers: {self.max_workers} (user cap)")
    
    def _get_allocated_cpu_cores(self):
        """Get the set of CPU core IDs available to this process"""
        affinity_cores = self._get_affinity_based_cpu_cores()
        if affinity_cores is not None:
            return affinity_cores
        # Fallback: assume we can use all logical cores up to our limit
        return set(range(self.available_cpus))

    def get_system_usage(self):
        """Get current resource usage (cached at 0.5s TTL to avoid blocking)"""
        now = time.time()
        if self._usage_cache and (now - self._usage_cache_time) < self._cache_ttl:
            return self._usage_cache

        mem = psutil.virtual_memory()

        try:
            all_cpu_usage = psutil.cpu_percent(interval=None, percpu=True)  # non-blocking
            available_cores = self._get_allocated_cpu_cores()
            cpu_per_core = [all_cpu_usage[i] for i in available_cores if i < len(all_cpu_usage)]
            cpu_percent = sum(cpu_per_core) / len(cpu_per_core) if cpu_per_core else 0
        except Exception:
            cpu_percent = psutil.cpu_percent(interval=None)
            cpu_per_core = None

        result = {
            'memory_percent': mem.percent,
            'memory_available_gb': mem.available / (1024**3),
            'cpu_percent': cpu_percent,
            'cpu_per_core': cpu_per_core,
        }
        self._usage_cache = result
        self._usage_cache_time = now
        return result

    def has_available_cpu(self, pending_workers=0):
        """Check if real-time resources allow a new job.

        pending_workers: number of workers spawned in the current batch that have
        not yet been measured by the OS.  Their estimated memory is accounted for
        as if already resident, preventing over-spawning within a single batch.

        Memory gate uses RSS budget: total child RSS + (pending+1) × estimate must
        fit within 80% of the SLURM allocation.  psutil.available is NOT used here
        because it includes reclaimable page cache and routinely shows 47GB "free"
        at 83% physical usage, making the gate fire far too late.
        """
        usage = self.get_system_usage()

        budget_gb = self.available_memory_gb * self.limits.mem_target_fraction
        total_rss = self._last_total_rss_gb
        projected = total_rss + (pending_workers + 1) * self.memory_per_job_gb
        if projected > budget_gb:
            logger.debug(
                f"RSS budget exceeded: {projected:.1f}GB projected "
                f"({total_rss:.1f}GB rss + {pending_workers+1} × {self.memory_per_job_gb:.2f}GB) "
                f"vs {budget_gb:.0f}GB budget"
            )
            return False

        # For non-SLURM: also gate on actual free memory.  The RSS projection above
        # uses the EWMA estimate, which can lag by 30-60 s during startup when workers
        # load large datasets.  This catches that window: require free memory to cover
        # one more worker's estimate PLUS the safety headroom before spawning.
        if not self.is_slurm:
            _free_gb = psutil.virtual_memory().available / (1024**3)
            _headroom_gb = self.available_memory_gb * (1 - self.limits.mem_target_fraction)
            if _free_gb < self.memory_per_job_gb + _headroom_gb:
                logger.debug(
                    f"Free memory floor: {_free_gb:.1f}GB free, need "
                    f"{self.memory_per_job_gb:.1f}GB (estimate) + {_headroom_gb:.1f}GB (headroom)"
                )
                return False

        if usage['cpu_percent'] > self.limits.cpu_max_usage * 100:
            logger.debug(f"CPU saturated: {usage['cpu_percent']:.1f}% > {self.limits.cpu_max_usage * 100:.0f}%")
            return False

        return True

    def get_actual_job_memory_gb(self) -> float:
        """Return RSS of this process + all children (throttled to mem_estimate_update_interval_s)."""
        now = time.time()
        if (now - self._last_mem_check_time) > self.limits.mem_estimate_update_interval_s:
            try:
                proc = psutil.Process()
                total_rss = proc.memory_info().rss
                for child in proc.children(recursive=True):
                    try:
                        total_rss += child.memory_info().rss
                    except psutil.NoSuchProcess:
                        pass
                self._cached_job_mem_gb = total_rss / (1024 ** 3)
                self._last_mem_check_time = now
            except Exception:
                pass  # keep previous cached value
        return self._cached_job_mem_gb

    def is_memory_pressure_elevated(self) -> bool:
        """Soft gate with hysteresis: pause feeding + backoff.

        Enters elevated when actual job RSS exceeds mem_elevated_fraction of allocation;
        exits only after RSS drops to mem_elevated_exit_fraction — prevents rapid
        oscillation when memory hovers near the boundary.
        """
        job_rss_gb = self.get_actual_job_memory_gb()
        alloc = self.available_memory_gb
        if self._in_elevated_state:
            self._in_elevated_state = job_rss_gb > alloc * self.limits.mem_elevated_exit_fraction
        else:
            self._in_elevated_state = job_rss_gb > alloc * self.limits.mem_elevated_fraction
        return self._in_elevated_state

    def is_memory_pressure_critical(self) -> bool:
        """Hard emergency brake: OOM recovery if machine OR Slurm allocation is critical."""
        if psutil.virtual_memory().percent > self.limits.mem_critical_machine_fraction * 100:
            return True
        if self.is_slurm:
            if self.get_actual_job_memory_gb() > self.available_memory_gb * self.limits.mem_critical_alloc_fraction:
                return True
        return False

    def update_memory_estimate(self, alive_workers: int) -> None:
        """EWMA-update memory_per_job_gb from live child-process RSS measurements.

        Throttled to mem_estimate_update_interval_s; requires min_alive_for_estimate
        workers to avoid noisy readings during ramp-up.  Alpha is intentionally low
        (0.05) to track sustained shifts (lighter job family mid-run) without reacting
        to transient dips (workers idle between jobs).  Bidirectional so the spawn gate
        self-calibrates throughout the grid search.
        """
        if alive_workers == 0:
            # All workers dead — reset stale total so the spawn gate isn't blocked by
            # the peak RSS of the previous worker cohort.
            self._last_total_rss_gb = 0.0
            return
        if alive_workers < self.limits.min_alive_for_estimate:
            return
        _now = time.time()
        if _now - self._obs_last_update < self.limits.mem_estimate_update_interval_s:
            return
        self._obs_last_update = _now

        worker_rss_bytes = 0
        try:
            proc = psutil.Process()
            for child in proc.children(recursive=True):
                try:
                    worker_rss_bytes += child.memory_info().rss
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
        except Exception:
            return
        if worker_rss_bytes <= 0:
            return
        self._last_total_rss_gb = worker_rss_bytes / (1024**3)
        observed = self._last_total_rss_gb / alive_workers
        # Skip early-startup readings — a tiny observed value poisons the EWMA downward,
        # causing the spawn gate to allow far more workers than are safe.
        if observed < self._initial_memory_per_job_gb * self.limits.startup_rss_min_fraction:
            return
        if self._obs_mem_ewma is None:
            self._obs_mem_ewma = observed
        else:
            # Asymmetric EWMA: adapt fast when observed > estimate (OOM risk), slow
            # when observed < estimate (conservative — don't shed workers on a transient dip).
            # Startup phase uses the fast alpha in both directions until the estimate converges.
            if self._obs_n_samples < self.limits.cpu_ewma_startup_n:
                α = self.limits.cpu_ewma_startup_alpha
            elif observed > self._obs_mem_ewma:
                α = self.limits.cpu_ewma_alpha_up
            else:
                α = self.limits.cpu_ewma_alpha
            self._obs_mem_ewma = α * observed + (1 - α) * self._obs_mem_ewma
        self._obs_n_samples += 1
        prev = self.memory_per_job_gb
        self.memory_per_job_gb = max(self.limits.min_memory_per_job_gb, self._obs_mem_ewma)
        if abs(self.memory_per_job_gb - prev) > 0.1:
            logger.info(
                f"Memory estimate updated: {prev:.2f}GB → {self.memory_per_job_gb:.2f}GB/worker "
                f"(observed {observed:.2f}GB, ewma={self._obs_mem_ewma:.2f}GB, alive={alive_workers})"
            )

    def handle_oom(self):
        """Handle out-of-memory error — double estimate so the spawn gate tightens."""
        prev = self.memory_per_job_gb
        self.memory_per_job_gb = self.memory_per_job_gb * 2.0
        self._obs_mem_ewma = self.memory_per_job_gb
        logger.warning(f"OOM detected: increased memory estimate {prev:.1f}GB → {self.memory_per_job_gb:.1f}GB")
    
    def get_status(self):
        """Get current system resource status string.

        Slurm: shows alloc % (job RSS / allocation) AND machine % side-by-side so the
        user can tell immediately whether to raise the Slurm memory request.
        Non-Slurm: shows machine % with free GB.
        """
        usage = self.get_system_usage()
        mem = psutil.virtual_memory()

        cpu_str = f"{usage['cpu_percent']:.0f}% cpu"
        if self.logical_cpus > self.physical_cpus:
            cpu_str += " (logi)"

        if self.is_slurm:
            job_rss_gb = self._cached_job_mem_gb
            alloc_pct = (job_rss_gb / self.available_memory_gb * 100) if self.available_memory_gb else 0
            mem_str = (f"alloc {alloc_pct:.0f}% ({job_rss_gb:.1f}/{self.available_memory_gb:.0f}GB) | "
                       f"machine {mem.percent:.0f}%")
        else:
            mem_str = f"mem {mem.percent:.0f}% ({usage['memory_available_gb']:.1f}GB free)"

        status = f"{'SLURM' if self.is_slurm else 'SYSTEM'}: {mem_str} | {cpu_str}"
        if usage.get('cpu_per_core'):
            status += f" | {grouped_bar_graph(usage['cpu_per_core'])}"
        return status
    

class GPUJobResourceManager:
    def __init__(self, memory_per_job_gb=2.0, buffer_gb=0.1, initial_concurrency=2, limits=None):
        self.limits = limits or ResourceLimits()
        self.memory_per_job_gb = memory_per_job_gb
        self.buffer_gb = buffer_gb
        self.initial_concurrency = initial_concurrency

        self.gpu_metrics = defaultdict(lambda: {
            'memory_util':  {'history': deque(maxlen=10), 'ewma': 0.0},
            'compute_util': {'history': deque(maxlen=10), 'ewma': 0.0},
            'last_sample_time': 0,
        })
        self.sample_interval = 1.0
        self.ewma_alpha = self.limits.gpu_ewma_alpha

        # Raw memory read cache — avoids a pynvml call on every dispatch decision.
        # TTL is half the EWMA sample_interval so the cache is always fresher than
        # the smoothed metrics but still prevents a pynvml call per scheduler tick.
        self._mem_cache: dict = {}             # gpu_id -> {'total', 'used', 'free', 'ts'}
        self._mem_cache_ttl: float = self.limits.mem_cache_ttl_s * 2  # = sample_interval * 0.5

        # Thresholds
        self.memory_util_threshold = self.limits.gpu_memory_util_max
        self.compute_lo_threshold = self.limits.gpu_compute_lo
        self.compute_hi_threshold = self.limits.gpu_compute_hi
        self.gpu_saturated: dict = {}      # per-gpu hysteresis flag (set in _init_pynvml)
        
        self._init_slurm_gpus()
        self._init_pynvml()

    def _init_slurm_gpus(self):
        self.available_gpus = []
        self._physical_id = {}
        
        slurm_gpus = environ.get('CUDA_VISIBLE_DEVICES')
        if slurm_gpus:
            try:
                for torch_idx, phys in enumerate(int(g.strip()) for g in slurm_gpus.split(',')):
                    self.available_gpus.append(torch_idx)
                    self._physical_id[torch_idx] = phys
            except ValueError:
                raise RuntimeError(f"Cannot parse CUDA_VISIBLE_DEVICES: {slurm_gpus}")

        if not self.available_gpus:
            if False:
                self.available_gpus = list(range(0))
                self._physical_id = {i: i for i in self.available_gpus}
            else:
                raise RuntimeError("No GPUs available")

    def _handle(self, gpu_id):
        return self._pynvml.nvmlDeviceGetHandleByIndex(self._physical_id[gpu_id])

    def _init_pynvml(self):
        """Initialize NVIDIA ML library and GPU tracking"""
        import pynvml  # lazy: only needed when GPU jobs are actually used
        self._pynvml = pynvml
        try:
            self._pynvml.nvmlInit()
        except Exception as e:
            raise RuntimeError(f"pynvml initialization failed: {e}")
        
        failed_gpus = []
        for gpu_id in self.available_gpus:
            try:
                handle = self._handle(gpu_id)
                device_name = self._pynvml.nvmlDeviceGetName(handle)
                if isinstance(device_name, bytes):
                    device_name = device_name.decode('utf-8')
                mem_info = self._pynvml.nvmlDeviceGetMemoryInfo(handle)
                
                logger.info(f"GPU {gpu_id}: {device_name}, "
                           f"Total Memory: {mem_info.total/(1024**3):.2f}GB")
                
                self.gpu_saturated[gpu_id] = False
            except Exception as e:
                logger.warning(f"Failed to initialize GPU {gpu_id}: {e}")
                failed_gpus.append(gpu_id)

        for gpu_id in failed_gpus:
            self.available_gpus.remove(gpu_id)

        if not self.available_gpus:
            raise RuntimeError("No GPUs could be initialized")

    def _update_ewma_metric(self, gpu_id, key, value):
        """Apply EWMA to a named metric slot and append to history."""
        m = self.gpu_metrics[gpu_id][key]
        m['history'].append(value)
        m['ewma'] = value if m['ewma'] == 0.0 else (
            self.ewma_alpha * value + (1 - self.ewma_alpha) * m['ewma']
        )

    def _get_gpu_memory_info(self, gpu_id):
        """Return GPU memory figures and the current memory-util EWMA.

        Raw pynvml reads are cached at _mem_cache_ttl (= sample_interval/2) so the
        scheduler loop never blocks on a pynvml call every tick.  EWMA updates still
        fire at sample_interval — they just use cached values when the cache is fresh.
        """
        metrics = self.gpu_metrics[gpu_id]
        current_time = time.time()
        cache = self._mem_cache.get(gpu_id)
        try:
            # Raw memory read: use cache when fresh, otherwise call pynvml and repopulate.
            if cache is not None and (current_time - cache['ts']) < self._mem_cache_ttl:
                total, used, free = cache['total'], cache['used'], cache['free']
            else:
                handle = self._handle(gpu_id)
                mem_info = self._pynvml.nvmlDeviceGetMemoryInfo(handle)
                total, used, free = mem_info.total, mem_info.used, mem_info.free
                self._mem_cache[gpu_id] = {'total': total, 'used': used, 'free': free, 'ts': current_time}

            # EWMA update: fires at sample_interval regardless of whether cache was hit.
            if current_time - metrics['last_sample_time'] >= self.sample_interval:
                mem_frac = used / total if total > 0 else 0
                self._update_ewma_metric(gpu_id, 'memory_util', mem_frac)
                try:
                    handle = self._handle(gpu_id)
                    util = self._pynvml.nvmlDeviceGetUtilizationRates(handle)
                    self._update_ewma_metric(gpu_id, 'compute_util', util.gpu / 100.0)
                except Exception:
                    pass  # some GPUs don't support utilisation rates; EWMA stays at 0
                metrics['last_sample_time'] = current_time

            return total, used, free, metrics['memory_util']['ewma']
        except Exception as e:
            logger.error(f"Error getting GPU {gpu_id} memory info: {e}")
            return 0, 0, 0, 0

    def _has_memory_capacity(self, gpu_id):
        """Check if GPU has enough memory for another job."""
        try:
            _, _, mem_free, avg_util = self._get_gpu_memory_info(gpu_id)
            required_bytes = (self.memory_per_job_gb + self.buffer_gb) * 1024**3
            return avg_util <= self.memory_util_threshold and mem_free >= required_bytes
        except Exception as e:
            logger.error(f"Error checking GPU {gpu_id} memory: {e}")
            return False

    def _has_compute_capacity(self, gpu_id):
        """Hysteretic compute-utilisation gate: accept below lo threshold, reject above hi."""
        avg_compute = self.gpu_metrics[gpu_id]['compute_util']['ewma']
        if self.gpu_saturated.get(gpu_id, False):
            if avg_compute < self.compute_lo_threshold:
                self.gpu_saturated[gpu_id] = False  # recovered
            else:
                return False                         # still saturated
        else:
            if avg_compute >= self.compute_hi_threshold:
                self.gpu_saturated[gpu_id] = True
                logger.debug(
                    f"GPU {gpu_id}: compute saturated ({avg_compute:.0%}), pausing dispatch"
                )
                return False
        return True

    def _can_allocate_on_gpu(self, gpu_id):
        """Check if we can allocate a job to this GPU."""
        return (gpu_id in self.available_gpus
                and self._has_memory_capacity(gpu_id)   # samples metrics as side-effect
                and self._has_compute_capacity(gpu_id))  # reads fresh EWMA from dict

    def handle_oom(self, gpu_id):
        """Handle out-of-memory error"""
        logger.warning(f"OOM on GPU {gpu_id}")

        # Clear stale metrics and reset hysteresis
        metrics = self.gpu_metrics[gpu_id]
        for key in ('memory_util', 'compute_util'):
            metrics[key]['history'].clear()
            metrics[key]['ewma'] = 0.0
        self.gpu_saturated[gpu_id] = False

        # Invalidate raw memory cache so the next dispatch gets a fresh pynvml read.
        self._mem_cache.pop(gpu_id, None)


    def get_status(self):
        """Get current GPU resource status (reads from cache, no direct pynvml call)."""
        status = []
        for gpu_id in self.available_gpus:
            try:
                _, _, mem_free, _ = self._get_gpu_memory_info(gpu_id)
                avg_mem = self.gpu_metrics[gpu_id]['memory_util']['ewma'] * 100
                avg_compute = self.gpu_metrics[gpu_id]['compute_util']['ewma'] * 100
                status.append(
                    f"GPU {gpu_id}: {mem_free/(1024**3):.1f}GB free "
                    f"(avg: {avg_mem:.0f}% mem, {avg_compute:.0f}% compute)"
                )
            except Exception:
                status.append(f"GPU {gpu_id}: unavailable")
        return " | ".join(status)


class BeeRouter:
    def __init__(self, exploration_rate: float = 0.1):
        self.exploration_rate = exploration_rate
        self._last_time   = {}  # device → elapsed seconds of last completion
        self._last_update = {}  # device → unix timestamp of last completion

    def record_completion(self, device: str, elapsed: float):
        self._last_time[device]   = elapsed
        self._last_update[device] = time.time()

    def rank_devices(self, devices: list, default_t: float = 60.0) -> list:
        """Epsilon-greedy device ranking: exploit fastest device most of the time,
        explore randomly with probability exploration_rate to avoid getting stuck
        if a device slows down or recovers after OOM backoff.
        """
        if random.random() < self.exploration_rate:
            shuffled = list(devices)
            random.shuffle(shuffled)
            return shuffled
        return sorted(devices,
                      key=lambda d: self._get_effective_weight(d, default_t),
                      reverse=True)

    def _get_effective_weight(self, device: str, default_t: float) -> float:
        last_t = self._last_time.get(device)
        if last_t is None:
            return 1.0 / max(default_t, 1e-6)
        silence = max(time.time() - self._last_update.get(device, 0), 1e-6)
        # Penalise proportionally once silence exceeds last completion time
        penalty = min(1.0, last_t / silence)
        return (1.0 / max(last_t, 1e-6)) * penalty




class ComputeJobResourceManager:
    def __init__(self, cpu_memory_per_job_gb=2.0, cpu_cores_per_job=1, cpu_max_workers=None,
                 exploration_rate: float = 0.1, gpu_memory_per_job_gb=None, limits=None):  # gpu_memory_per_job_gb ignored (CPU-only)
        self.limits = limits or ResourceLimits()
        self.cpu_manager = CPUJobResourceManager(
            memory_per_job_gb=cpu_memory_per_job_gb,
            cores_per_job=cpu_cores_per_job,
            max_workers=cpu_max_workers,
            limits=self.limits,
        )

        environ['CUDA_VISIBLE_DEVICES'] = ''  # CPU-only: hide GPUs
        self.use_gpu = False
        self.gpu_manager = None
        if self.use_gpu:
            self.gpu_manager = GPUJobResourceManager(
                memory_per_job_gb=gpu_memory_per_job_gb,
                limits=self.limits,
            )

        # Pre-build device objects once — reused in hot path instead of allocating per call
        self._cpu_device = 'cpu'
        self._gpu_devices = (
            {gpu_id: f'cuda:{gpu_id}' for gpu_id in self.gpu_manager.available_gpus}
            if self.use_gpu else {}
        )

        self.bee_router = BeeRouter(exploration_rate=exploration_rate)

        logger.info(f"Mode: {'GPU+CPU' if self.use_gpu else 'CPU only'}")

    def try_allocate_for_device(self, device):
        """Check if resources are available for a job on a specific device."""
        try:
            return self.can_spawn_worker(device, pending_workers=0)
        except Exception:
            return False

    def can_spawn_worker(self, device: str, pending_workers: int = 0) -> bool:
        """Spawn gate: CPU check (all workers) + GPU check (CUDA workers only).

        pending_workers: workers spawned in this round not yet visible to the OS.
        Their estimated memory is pre-subtracted from available RAM before the
        CPU threshold is evaluated.
        """
        if not self.cpu_manager.has_available_cpu(pending_workers=pending_workers):
            return False
        if device.startswith('cuda') and self.gpu_manager is not None:
            if not self.gpu_manager._can_allocate_on_gpu(device.index):
                return False
        return True

    def handle_oom(self, device):
        if device.startswith('cuda'):
            gpu_id = device.index
            self.gpu_manager.handle_oom(gpu_id)
        else:
            self.cpu_manager.handle_oom()

    def is_memory_pressure_elevated(self) -> bool:
        return self.cpu_manager.is_memory_pressure_elevated()

    def is_memory_pressure_critical(self) -> bool:
        return self.cpu_manager.is_memory_pressure_critical()

    @property
    def max_concurrent(self):
        n = self.cpu_manager.available_cpus // self.cpu_manager.cores_per_job
        if self.cpu_manager.max_workers is not None:
            n = min(n, self.cpu_manager.max_workers)
        return n

    @property
    def cores_per_job(self):
        return self.cpu_manager.cores_per_job