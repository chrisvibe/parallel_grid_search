import collections
import contextlib
from pathlib import Path
from time import sleep, time
import queue
import random
import numpy as np  # Must be imported before torch to avoid MKL/OpenMP threading conflicts in multiprocessing
import multiprocessing as tmp
tmp.set_start_method("spawn", force=True)  # match torch.multiprocessing default for juliacall safety
import psutil
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
import threading
from typing import Tuple, Callable, List, Dict, Any
from project.parallel_grid_search.code.parallel_utils import SQLiteLock, GridSearchDB, JobGenerator, ComputeJobResourceManager, ResourceLimits, JobInterface, DATASET_LOCK_KEY, RUN
import os
from os import environ
import signal
import tempfile



import socket

# --- HPC node-local tmp directory ---
# On shared-filesystem HPC clusters, /tmp may be shared or nonexistent.
# Redirect tempfile and torch.compile cache to a node-local directory
# to prevent cross-job interference (stale locks, cache collisions).
def _setup_node_tmp():
    """Set node-local tmp and inductor cache dirs to prevent cross-job interference on HPC."""
    node = environ.get("SLURMD_NODENAME") or socket.gethostname()
    tmp_dir = Path(f'/tmp/{node}')
    tmp_dir.mkdir(exist_ok=True)
    tempfile.tempdir = str(tmp_dir)
_setup_node_tmp()

def _apply_mp_workarounds():
    """Patch multiprocessing to avoid semaphore resource-tracker bugs with spawn mode.

    Two patches applied:
    1. resource_tracker: skip semaphores — the tracker can't distinguish inherited
       vs. created semaphores and unlinks queues still in use by the parent, causing
       FileNotFoundError on the next worker spawn.
    2. SemLock._cleanup: suppress the matching FileNotFoundError in parent finalizers
       when the child already unlinked the semaphore on exit.

    TODO: Remove when https://github.com/python/cpython/issues/82300 is fixed.
    """
    from multiprocessing import resource_tracker as _rt

    _rt_register = _rt.register
    _rt_unregister = _rt.unregister

    def _register_except_semaphore(name, rtype):
        if rtype != "semaphore":
            _rt_register(name, rtype)

    def _unregister_except_semaphore(name, rtype):
        if rtype != "semaphore":
            _rt_unregister(name, rtype)

    _rt.register = _register_except_semaphore
    _rt.unregister = _unregister_except_semaphore

    from multiprocessing import synchronize as _mp_sync

    @staticmethod
    def _safe_sem_cleanup(name):
        pass  # Never unlink — parent owns these; OS/cgroup cleans up on job exit.

    _mp_sync.SemLock._cleanup = _safe_sem_cleanup

_apply_mp_workarounds()

# --- TORCH-AWARE START METHOD ---
if tmp.get_start_method(allow_none=True) is None:
    tmp.set_start_method('spawn')

import logging
logger = logging.getLogger(__name__)


def _cuda_health_check(device):
    """Quick CUDA health check — returns True if device is usable, False otherwise."""
    try:
        del test
        return True
    except Exception as e:
        logger.error(f"CUDA health check FAILED on {device}: {type(e).__name__}: {e}", exc_info=True)
        return False


def _make_error_result(job, device, pid, start_time, error_type, error_msg):
    return {
        'status': 'error',
        'error_type': error_type,
        'error': error_msg,
        'job': job,
        'device': device,
        'worker_pid': pid,
        'start_time': start_time,
        'stop_time': time(),
    }


@contextlib.contextmanager
def _signal_guard(handlers: dict):
    """Temporarily install signal handlers (sig → callable), restoring originals on exit."""
    saved = {sig: signal.signal(sig, handler) for sig, handler in handlers.items()}
    try:
        yield
    finally:
        for sig, orig in saved.items():
            signal.signal(sig, orig)


@contextlib.contextmanager
def _job_timeout_guard(job, timeout: float):
    """SIGALRM-based per-job deadline. Raises TimeoutError if timeout elapses."""
    def _handler(signum, frame):
        raise TimeoutError(f"Job {job.i}-{job.j} exceeded job_timeout ({timeout:.0f}s)")
    with _signal_guard({signal.SIGALRM: _handler}):
        signal.alarm(int(timeout))
        try:
            yield
        finally:
            signal.alarm(0)


def worker_function(job_queue, result_queue, cores_per_job, priority, device,
                    idle_timeout: float = 120.0, job_timeout: float = 3600.0,
                    max_jobs: int | None = None):
    """Worker function - no shared state, just queues.

    Workers live until the queue is empty for idle_timeout seconds (starvation exit)
    or they receive a None sentinel (soft kill / scale-down).
    idle_timeout is the laziness knob: longer = more resistant to brief W dips.
    job_timeout: per-job SIGALRM deadline in seconds; raises TimeoutError if exceeded.
    max_jobs: recycle the worker after this many completed jobs (resets allocator
    fragmentation). None / 0 = unlimited (current default behaviour).
    """
    # Belt-and-suspenders: cancel any SemLock._cleanup finalizers registered during
    # unpickling.  In spawn mode the module-level no-op patch may not have been applied
    # before SemLock.__setstate__ ran, leaving finalizers that would call sem_unlink on
    # exit — deleting the parent-owned semaphore and breaking subsequent worker spawns.
    from multiprocessing.util import _finalizer_registry as _fr
    for _fin in list(_fr.values()):
        if getattr(getattr(_fin, '_callback', None), '__qualname__', '').endswith('._cleanup'):
            _fin.cancel()

    pid = os.getpid()
    p = psutil.Process()


    try:
        p.nice(priority)
    except (psutil.AccessDenied, PermissionError):
        pass

    jobs_done = 0
    if max_jobs is not None:
        # Smear recycling across [80%, 100%] of max_jobs so workers don't all die together.
        _jitter = int(max_jobs * 0.2)
        max_jobs = random.randint(max_jobs - _jitter, max_jobs)
    while True:
        try:
            job_data = job_queue.get(timeout=idle_timeout)
        except queue.Empty:
            logger.debug(f"Worker PID {pid}: idle timeout on {device}, exiting")
            break
        if job_data is None or job_data.get('job') is None:
            break  # sentinel from softly_kill_all_workers or scale-down

        job = job_data['job']
        start_time = time()
        logger.debug(f"Worker PID {pid}: starting job {job.i}-{job.j} on {device}")
        result_queue.put({'status': 'started', 'key': (job.i, job.j), 'pickup_time': start_time, 'worker_pid': pid})

        try:
            with _job_timeout_guard(job, job_timeout):
                result = job.run(device)
            result.update({
                'job': job,
                'device': device,
                'worker_pid': pid,
                'start_time': start_time,
                'stop_time': time(),
            })
        except Exception as e:
            logger.error(f"Worker PID {pid}: UNHANDLED exception in job {job.i}-{job.j} on {device}: {e}", exc_info=True)
            if device.startswith('cuda'):
                try:
                    logger.error(f"Worker PID {pid}: CUDA memory state on {device}:\n{mem_summary}")
                except Exception:
                    pass
                cuda_ok = _cuda_health_check(device)
                logger.error(f"Worker PID {pid}: CUDA health after crash: {'OK' if cuda_ok else 'FAILED'}")
            result = _make_error_result(job, device, pid, start_time,
                                        'worker_crash', f'Unhandled {type(e).__name__}: {e}')
        finally:
            job.locks = {}  # always clear locks before result leaves the worker

        try:
            result_queue.put(result)
        except Exception as e:
            logger.error(f"Result serialization failed for job {job.i}-{job.j}: {e}")
            try:
                result_queue.put(
                    _make_error_result(job, device, pid, start_time,
                                       'serialization', f'Result serialization failed: {type(e).__name__}: {e}')
                )
            except Exception as e2:
                logger.critical(
                    f"Even fallback result failed for job {job.i}-{job.j}: {e2}"
                )

        if hasattr(result['job'], 'cleanup'):
            result['job'].cleanup()

        logger.debug(f"Worker completed job {result['job'].i}-{result['job'].j} on {device}")
        jobs_done += 1
        if max_jobs is not None and jobs_done >= max_jobs:
            logger.debug(f"Worker PID {pid}: recycling after {jobs_done} jobs on {device}")
            break
    logger.debug(f"Worker PID {pid}: exiting on {device}")


def worker_function_cpu(*args, **kwargs):
    environ['CUDA_VISIBLE_DEVICES'] = ''  # hide GPU
    return worker_function(*args, **kwargs)

def drain_queue(queue):
    while not queue.empty():
        try:
            queue.get_nowait()
        except:
            break


class LazyWorkerPool:
    """Worker pool with supervisor pattern - parent owns all state.

    Workers self-terminate when the job queue is empty.  The pool
    auto-scales up (pool-internally) when jobs are submitted: if J jobs
    are waiting in the queue, up to max(1, J // 2) new workers are
    spawned, subject to the resource gate supplied via *can_spawn_fn*.
    There is no explicit scale-down — workers die naturally.
    """

    def __init__(self, max_workers, cores_per_job, process_priority=5, device='cpu',
                 can_spawn_fn=None, idle_timeout: float = 120.0, job_timeout: float = 3600.0,
                 spawn_delay: float = 30.0, max_scale: int = 1):
        self.device = device
        self.max_workers = max_workers
        self.cores_per_job = cores_per_job
        self.process_priority = process_priority
        self.can_spawn_fn = can_spawn_fn or (lambda pending=0: True)
        self.idle_timeout = idle_timeout
        self.job_timeout = job_timeout
        self.spawn_delay = spawn_delay  # min seconds between scale events (laziness knob)
        self.max_scale = max_scale      # max workers to spawn/kill per scale event
        self._last_scale_time: float = 0.0
        self.worker_job_queue = tmp.Queue()
        self.result_queue = tmp.Queue()

        # Parent-owned state — no SyncManager needed
        self.workers = dict()
        self._alive_count = 0   # maintained by _spawn_one_worker / _cleanup_dead_workers
        self._enabled = True
        # GPU starts at 1 (AIMD grows it); CPU starts uncapped at max_workers
        self._concurrency_limit = 1 if self.device.startswith('cuda') else max_workers
        self._sigkill_detected = False
        self._in_backoff = False
        self._last_recovery_time: float = 0.0

        self.name = device + '_pool'
        logger.debug(f"{self.name} initialized (max_workers={max_workers})")

    @property
    def alive_workers(self):
        return self._alive_count

    @property
    def backlog(self):
        return self.worker_job_queue.qsize()

    @property
    def satisfied(self):
        return self._alive_count >= 1 and self.backlog > self._alive_count * 2

    @property
    def enabled(self):
        return self._enabled

    def enable(self):
        self._enabled = True

    def disable(self):
        self._enabled = False

    @property
    def current_concurrency_limit(self):
        return self._concurrency_limit

    @current_concurrency_limit.setter
    def current_concurrency_limit(self, value):
        self._concurrency_limit = min(max(1, int(value)), self.max_workers)

    # ------------------------------------------------------------------
    # Internal scaling
    # ------------------------------------------------------------------

    def _cleanup_dead_workers(self):
        """Remove exited workers from the tracking dict, logging their exit codes."""
        for pid, process in list(self.workers.items()):
            if not process.is_alive():
                process.join(timeout=1)
                exit_code = process.exitcode
                if exit_code is not None and exit_code < 0:
                    try:
                        sig_name = signal.Signals(-exit_code).name
                    except (ValueError, AttributeError):
                        sig_name = f"signal {-exit_code}"
                    if exit_code == -signal.SIGKILL:
                        logger.critical(f"{self.name}: Worker PID {pid} killed by Linux OOM killer (SIGKILL)")
                        self._sigkill_detected = True
                    else:
                        logger.error(f"{self.name}: Worker PID {pid} KILLED by {sig_name} (exit code {exit_code})")
                elif exit_code is not None and exit_code > 0:
                    logger.error(f"{self.name}: Worker PID {pid} exited with error code {exit_code}")
                else:
                    # exit_code == 0: expected self-termination (queue went empty)
                    logger.debug(f"{self.name}: Worker PID {pid} exited normally (queue empty)")
                self._alive_count -= 1
                del self.workers[pid]

    def _spawn_one_worker(self):
        """Spawn a single new worker process."""
        target = worker_function if self.device.startswith('cuda') else worker_function_cpu
        max_jobs = int(os.environ.get('WORKER_MAX_JOBS', '0')) or None
        # Jitter idle_timeout per-worker so a cohort that all go idle together doesn't
        # all time out simultaneously — staggered deaths avoid the mass-death→stale-RSS
        # cliff even after the RSS reset fix.
        jittered_idle_timeout = self.idle_timeout * random.uniform(0.8, 1.2)
        p = tmp.Process(
            target=target,
            args=(self.worker_job_queue, self.result_queue, self.cores_per_job,
                  self.process_priority, self.device, jittered_idle_timeout, self.job_timeout,
                  max_jobs),
            daemon=True,
        )
        p.start()
        self.workers[p.pid] = p
        self._alive_count += 1

    def _scale_up(self):
        """Spawn up to max_scale workers if alive < target and spawn_delay has elapsed."""
        if self.backlog == 0:
            return 0
        if time() - self._last_scale_time < self.spawn_delay:
            return 0
        spawned = 0
        while spawned < self.max_scale:
            if self.alive_workers >= self.current_concurrency_limit:
                break
            if self.alive_workers >= self.max_workers:
                break
            if not self.can_spawn_fn(pending=spawned):
                break
            self._spawn_one_worker()
            spawned += 1
        if spawned > 0:
            self._last_scale_time = time()
            logger.debug(f"{self.name}: scale_up → spawned {spawned} (alive={self.alive_workers}, target={self.current_concurrency_limit})")
        return spawned

    def _scale_down(self):
        """Send up to max_scale sentinels if alive > target and spawn_delay has elapsed.

        Workers receive the sentinel after draining current queue work and exit.
        """
        if time() - self._last_scale_time < self.spawn_delay:
            return 0
        to_send = min(self.max_scale, self.alive_workers - self.current_concurrency_limit)
        if to_send <= 0:
            return 0
        for _ in range(to_send):
            self.worker_job_queue.put(None)
        self._last_scale_time = time()
        logger.debug(f"{self.name}: scale_down → sent {to_send} sentinels (alive={self.alive_workers}, target={self.current_concurrency_limit})")
        return to_send

    def scale(self):
        """Reconcile alive workers toward current_concurrency_limit.

        Always calls _cleanup_dead_workers first so stale _alive_count is
        corrected before spawn/kill decisions — this is the fix for GPU workers
        appearing alive but having actually died without being detected.
        """
        self._cleanup_dead_workers()
        alive = self.alive_workers
        W = self.current_concurrency_limit
        if alive < W:
            return self._scale_up()
        elif alive > W:
            return self._scale_down()
        return 0

    def backoff(self):
        """Multiplicative decrease on memory-pressure shock (AIMD-MD step).

        Reduces the concurrency limit rather than resetting to 1.  Idea is to converge
        to the true safe ceiling
        """
        if self._in_backoff:
            return False
        self._in_backoff = True
        prev = self._concurrency_limit
        self._concurrency_limit = max(1, (3 * prev) // 4)
        to_shed = max(0, self.alive_workers - self._concurrency_limit)
        for _ in range(to_shed):
            self.worker_job_queue.put(None)
        now = time()
        self._last_scale_time = now      # block rate-limited scale for one full period
        self._last_recovery_time = now   # recovery cooldown starts now
        logger.warning(
            f"{self.name}: memory pressure — backoff "
            f"(limit {prev} → {self._concurrency_limit}, shedding {to_shed} workers)"
        )
        return True

    def submit_job(self, job):
        if self._enabled:
            self.worker_job_queue.put({'job': job})
            return True
        else:
            return False

    def get_result(self, timeout=None):
        """Get a result from the result queue"""
        try:
            result = self.result_queue.get(timeout=timeout)
            return result
        except queue.Empty:
            return None
        except (ValueError, EOFError, OSError):
            logger.debug(f"{self.name}: Result queue closed")
            return None

    def shutdown(self):
            """Shutdown all workers and clean up resources"""
            logger.debug(f"{self.name}: Initiating shutdown")
            
            # 1. Stop accepting new jobs
            self._enabled = False
            
            # 2. Tell workers to exit and clear the queue
            self.softly_kill_all_workers()
            sleep(0.5)
            
            # 3. Force kill survivors
            self.hard_kill_all_workers()
            
            # 4. Manual Queue Cleanup
            # This prevents the 'feeder thread' from hanging your script at the end
            try:
                # Drain remaining items
                drain_queue(self.worker_job_queue)
                drain_queue(self.result_queue)
                
                # Close the queues
                self.worker_job_queue.close()
                self.result_queue.close()
                
                # Important: join_thread ensures the background IPC thread exits
                self.worker_job_queue.join_thread()
                self.result_queue.join_thread()
            except Exception as e:
                logger.debug(f"Queue cleanup error: {e}")
                
            logger.debug(f"{self.name}: shutdown complete")
    
    def softly_kill_all_workers(self):
        drain_queue(self.worker_job_queue)
        for _ in range(self.alive_workers):
            self.worker_job_queue.put(None)

    def hard_kill_all_workers(self):
        # SIGTERM all at once
        for proc in self.workers.values():
            proc.terminate()
        
        sleep(0.5)
        
        # SIGKILL any survivors
        for proc in self.workers.values():
            if proc.is_alive():
                proc.kill()
        
        # Wait for all at once
        for proc in self.workers.values():
            proc.join(timeout=1)
        
        self.workers.clear()
        self._alive_count = 0   # reset to match cleared dict

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.shutdown()


class ResourceAwareScheduler:
    """Simplified scheduler with centralized resource management"""


    def __init__(self, resource_manager: ComputeJobResourceManager, history: list, job_generator: JobGenerator, scheduler_loop_delay: int = .1, max_retries: int = 3, db: 'GridSearchDB | None' = None):
        self.resource_manager = resource_manager
        self.history = history
        self.running = False
        self.submission_complete = False
        self.scheduler_thread = None
        self.scheduler_loop_delay = scheduler_loop_delay

        # Worker pool — no manager needed
        def _make_can_spawn(device):
            return lambda pending=0: self.resource_manager.can_spawn_worker(device, pending_workers=pending)
        self.worker_pools = dict()
        _logical_available = self.resource_manager.max_concurrent
        _physical_cpus = self.resource_manager.cpu_manager.physical_cpus
        _effective_cores = max(1, min(_logical_available, _physical_cpus))
        cpu_pool = LazyWorkerPool(
            max_workers=_effective_cores,
            cores_per_job=self.resource_manager.cores_per_job,
            process_priority=15, can_spawn_fn=_make_can_spawn('cpu'),
            max_scale=max(1, int(_effective_cores * 0.1)),
        )
        self.worker_pools[cpu_pool.device] = cpu_pool
        if self.resource_manager.use_gpu:
            for gpu_id in self.resource_manager.gpu_manager.available_gpus:
                gpu_device = f'cuda:{gpu_id}'
                gpu_pool = LazyWorkerPool(
                    max_workers=8,  # GPU parallelism is memory-bound; AIMD grows from 1 up to this cap
                    cores_per_job=self.resource_manager.cores_per_job,
                    process_priority=10, device=f'cuda:{gpu_id}',
                    can_spawn_fn=_make_can_spawn(gpu_device),
                    max_scale=1,
                )
                self.worker_pools[gpu_pool.device] = gpu_pool

        # At-least-once delivery: retry tracking
        self.job_generator = job_generator
        self.in_flight = {}                # (i,j) -> {'queued_time', 'pickup_time', 'device', 'worker_pid'}
        self._in_flight_lock = threading.RLock()  # guards in_flight across feeder + scheduler threads

        self._completed_count = 0          # jobs that completed successfully
        self.permanently_failed = set()    # (i,j) pairs that exceeded max retries
        self.retry_counts = {}             # (i,j) -> int
        self.max_retries = max_retries
        self.total_expected = len(job_generator)

        # Dispatch cap: upper safety ceiling on total jobs in flight (queued + running).
        # in_flight includes backlog sitting in pool queues, not just active workers.
        # Buffer = 3× total worker slots so the feeder (20ms cadence) never starves workers.
        # Worker process count is still gated independently by can_spawn_fn / max_workers.
        self.max_in_flight = sum(p.max_workers for p in self.worker_pools.values()) * 3

        self._last_status_time = 0.0
        self._last_cap_limit: dict = {}  # pool → last capped value, used to suppress duplicate log lines

        self.db = db
        self._pending_marks: list = []  # (i,j) pairs to mark done after next Parquet flush
        self._history_lock = threading.Lock()  # serialises the two-step append to history + _pending_marks

        self._start_job_submissions(job_generator)

    def running_jobs_per_device(self, device):
        """Jobs actively being processed: dispatched to a pool minus jobs still in its queue."""
        with self._in_flight_lock:
            if device is None:
                dispatched = len(self.in_flight)
                queued = sum(pool.backlog for pool in self.worker_pools.values())
                return max(0, dispatched - queued)
            dispatched = sum(1 for info in self.in_flight.values() if info['device'] == device)
        pool = self.worker_pools.get(device)
        queued = pool.backlog if pool else 0
        return max(0, dispatched - queued)

    @property
    def completed_count(self):
        return self._completed_count
        
    def _start_job_submissions(self, job_generator):
        """Start scheduler and feeder threads."""
        self.running = True
        self._job_iter = iter(job_generator)
        self._retry_queue = collections.deque()
        self.scheduler_thread = threading.Thread(target=self._scheduler_loop, daemon=True)
        self.scheduler_thread.start()
        self._feeder_thread = threading.Thread(target=self._feeder_loop, daemon=True)
        self._feeder_thread.start()
        logger.info("Scheduler started")
        return self.running
    

    def stop(self):
        """Stop the scheduler and clean up all resources"""
        if not self.running:
            return
        self.running = False
            
        logger.info("Initiating scheduler shutdown")
        
        # Wait for scheduler thread
        if self.scheduler_thread and self.scheduler_thread.is_alive():
            self.scheduler_thread.join(timeout=5)
        
        # Shutdown worker pools
        for device, pool in self.worker_pools.items():
            pool.shutdown()
        
        for p in tmp.active_children():
            logger.warning(f"Force terminating orphan process {p.pid}")
            p.terminate()
            p.join()
                
        logger.info("Scheduler shutdown complete")
        
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
       if exc_type is not None:
           logger.error(f"Exiting due to exception: {exc_type.__name__}: {exc_val}")
       self.stop()
       return False

    def _handle_job_completion(self, result):
        """Handle job completion — bookkeeping only; dispatching is done by _scheduler_loop."""
        job = result['job']
        device = result['device']
        key = (job.i, job.j)

        # Remove from in-flight tracking (this IS the resource release)
        with self._in_flight_lock:
            self.in_flight.pop(key, None)

        if result['status'] == 'completed':
            exec_time  = result['stop_time'] - result['start_time']
            self.resource_manager.bee_router.record_completion(device, exec_time)
            with self._history_lock:
                if 'history' in result:
                    self.history.append(result['history'])
                self._pending_marks.append((job.i, job.j))
            self._completed_count += 1
            logger.debug(f"Completed job {job.i}-{job.j} on {device} in {exec_time:.1f}s")

        elif result['status'] == 'error':
            error_msg = result.get('error', 'unknown')
            if result.get('error_type') == 'OOM':
                logger.error(f"OOM job {job.i}-{job.j} on device {device}")
                self._handle_oom_cleanup(device)
            else:
                logger.error(f"Error on job {job.i}-{job.j} on device {device}: {error_msg}")
            self._retry_job(key, error_msg)

        # Dispatch is handled by _scheduler_loop after all results for the tick are collected.

    def _handle_oom_cleanup(self, device):
        """OOM-specific cleanup only. Job retry handled by _retry_job."""
        pool = self.worker_pools[device]
        pool.disable()
        self.resource_manager.handle_oom(pool.device)
        pool.enable()
        logger.debug(f"{pool.name} - {pool.device} recovered from OOM")

    def _retry_job(self, key, reason):
        """Re-create and re-queue a failed/lost job, or mark permanently failed."""
        i, j = key
        count = self.retry_counts.get(key, 0) + 1
        self.retry_counts[key] = count

        if count > self.max_retries:
            logger.error(f"Job {i}-{j} permanently failed after {self.max_retries} retries: {reason}")
            self.permanently_failed.add(key)
            if self.db is not None:
                self.db.mark_done_batch([(i, j)])  # advance to done so it doesn't block compaction
            return

        logger.warning(f"Retrying job {i}-{j} (attempt {count}/{self.max_retries}): {reason}")
        job = self.job_generator.job_factory(
            i=i, j=j,
            total_configs=self.job_generator.total_configs,
            total_samples=self.job_generator.samples_per_config,
            locks=self.job_generator.locks,
        )
        self._retry_queue.append(job)


    def _collect_results(self, pool: 'LazyWorkerPool', timeout: float = 0.0) -> None:
        """Drain pool's result queue and process every result."""
        while True:
            result = pool.get_result(timeout=timeout)
            if result is None:
                break
            if result.get('status') == 'started':
                key = result['key']
                with self._in_flight_lock:
                    if key in self.in_flight:
                        self.in_flight[key]['pickup_time'] = result['pickup_time']
                        self.in_flight[key]['worker_pid'] = result.get('worker_pid')
            else:
                self._handle_job_completion(result)

    def _check_lost_jobs(self):
        """Detect and retry jobs whose worker pool died without returning a result."""
        now = time()

        for device, pool in self.worker_pools.items():
            if pool.alive_workers == 0 and pool.backlog == 0:
                # Drain before declaring jobs lost — a worker may have pushed a result
                # just before dying.
                self._collect_results(pool)
                with self._in_flight_lock:
                    device_jobs = [(k, v) for k, v in list(self.in_flight.items()) if v['device'] == device]
                for key, info in device_jobs:
                    with self._in_flight_lock:
                        self.in_flight.pop(key, None)
                    age = now - info['queued_time']
                    logger.error(
                        f"Job {key[0]}-{key[1]} LOST on {device}: pool has 0 alive workers "
                        f"(job age: {age:.0f}s)"
                    )
                    self._retry_job(key, f'device pool dead ({device})')

            # Detect jobs whose assigned worker died while other workers are still alive.
            # _cleanup_dead_workers (called via pool.scale every 20ms in the feeder thread)
            # removes dead PIDs from pool.workers promptly — alive_pids reflects current state.
            alive_pids = set(pool.workers.keys())
            with self._in_flight_lock:
                orphaned = [
                    (k, v) for k, v in list(self.in_flight.items())
                    if v['device'] == device
                    and v.get('worker_pid') is not None
                    and v['worker_pid'] not in alive_pids
                ]
            if orphaned:
                self._collect_results(pool)  # drain defensively before declaring jobs lost
                for key, info in orphaned:
                    with self._in_flight_lock:
                        if key not in self.in_flight:
                            continue  # completed during drain above
                        self.in_flight.pop(key)
                    age = now - info['queued_time']
                    logger.error(
                        f"Job {key[0]}-{key[1]} ORPHANED on {device}: "
                        f"worker PID {info['worker_pid']} died (job age: {age:.0f}s)"
                    )
                    self._retry_job(key, f'worker died ({device})')
    
    def _feed_pools(self):
        """Kanban fill: top up every pool queue to 2 * current_concurrency_limit.

        Pools are shuffled so no device starves the others when
        scheduler_job_queue runs low. _try_scale_up (called after filling)
        always invokes _cleanup_dead_workers, keeping _alive_count accurate.
        """
        if not self.running:
            return

        # Back-pressure: when RAM is running low, let existing workers drain before
        # enqueueing more work. Still run _scale_pools for dead-worker cleanup.
        if self.resource_manager.is_memory_pressure_elevated():
            self._scale_pools()
            return

        pools = list(self.worker_pools.values())
        devices = [p.device for p in pools]
        known_times = [self.resource_manager.bee_router._last_time[d]
                       for d in devices if d in self.resource_manager.bee_router._last_time]
        default_t = max(known_times, default=60.0)
        device_to_pool = {p.device: p for p in pools}
        pools = [device_to_pool[d]
                 for d in self.resource_manager.bee_router.rank_devices(devices, default_t)]

        # Sync AIMD limits to GPU pools before spawning decisions
        for pool in pools:
            if pool.device.startswith('cuda'):
                pool.current_concurrency_limit = (
                    self.resource_manager.bee_router._concurrency_limit.get(pool.device, 1)
                )

        global_budget = max(0, self.max_in_flight - len(self.in_flight))

        for pool in pools:
            if not pool.enabled:
                continue
            if pool.device.startswith('cuda'):
                # GPU queue filling: only gate on GPU memory, not CPU load.
                # CPU load is the right gate for *worker spawning* (via can_spawn_fn),
                # but applying it here too starves GPU queues whenever CPU is saturated.
                if not self.resource_manager.gpu_manager._can_allocate_on_gpu(pool.device.index):
                    continue
            # CPU pool: memory pressure is already gated at the top of _feed_pools.
            # Don't gate CPU queue filling on cpu_load — same reason as GPU: it starves
            # workers when all workers are busy and CPU is saturated by fast jobs.

            target = 2 * pool.current_concurrency_limit
            needed = min(max(0, target - pool.backlog), global_budget)

            for _ in range(needed):
                if self._retry_queue:
                    job = self._retry_queue.popleft()
                elif not self.submission_complete:
                    try:
                        job = next(self._job_iter)
                    except StopIteration:
                        self.submission_complete = True
                        break
                else:
                    break
                pool.submit_job(job)
                with self._in_flight_lock:
                    self.in_flight[(job.i, job.j)] = {
                        'queued_time': time(), 'pickup_time': None, 'device': pool.device, 'worker_pid': None
                    }
                global_budget -= 1

        self._scale_pools()

    def _scale_pools(self):
        """Adjust each pool toward its W target, rate-limited by spawn_delay.

        Sets spawn_delay from avg job time (laziness adapts to workload speed)
        then delegates all scale logic — including dead-worker cleanup — to pool.scale().

        When memory pressure is elevated, calls pool.backoff() to shed workers
        immediately. When pressure drops the grow branch handles scale-up uniformly
        whether the pool is recovering from backoff or growing a fresh budget ceiling.
        _in_backoff stays True during grow-back as a dedup guard; it clears when
        the limit reaches effective_max.
        """
        # Update per-job memory estimate from live measurements so pressure gates
        # self-calibrate without needing the right cpu_memory_per_job_gb at startup.
        _cpu_pool = next((p for p in self.worker_pools.values() if p.device == 'cpu'), None)
        cpu_budget_limit: dict = {}  # pool → budget concurrency ceiling
        if _cpu_pool is not None:
            cpu_mgr = self.resource_manager.cpu_manager
            cpu_mgr.update_memory_estimate(_cpu_pool.alive_workers)
            # Compute RSS budget ceiling once per _scale_pools call.
            # Used as the effective max for both proactive shedding and budget growth
            # so the two control paths agree on the target and don't fight each other.
            # SLURM: use the static allocation (it's a hard quota).
            # Non-SLURM: use rss + available so the ceiling falls as workers load data,
            # triggering shedding before the next OOM even if EWMA is still converging.
            if cpu_mgr.is_slurm:
                budget_gb = cpu_mgr.available_memory_gb * self.resource_manager.limits.mem_target_fraction
            else:
                _free_gb = psutil.virtual_memory().available / (1024**3)
                budget_gb = (cpu_mgr._last_total_rss_gb + _free_gb) * self.resource_manager.limits.mem_target_fraction
            # Use max(ewma, observed) so the ceiling reacts immediately when actual RSS
            # exceeds the EWMA — prevents the EWMA lag from allowing too many workers
            # during startup and causing the elevated pressure gate to fire reactively.
            _alive = _cpu_pool.alive_workers
            _observed_per_job = (cpu_mgr._last_total_rss_gb / _alive) if _alive > 0 else 0.0
            _effective_per_job = max(cpu_mgr.memory_per_job_gb, _observed_per_job)
            cpu_budget_limit[_cpu_pool] = max(1, int(budget_gb / _effective_per_job))

        pressure = self.resource_manager.is_memory_pressure_elevated()
        for pool in self.worker_pools.values():
            avg_t = self.resource_manager.bee_router._last_time.get(pool.device, None)
            if avg_t is not None:
                # Floor of 5s instead of 30s: fast jobs (avg ~5s) were taking
                # 300s to reach full worker count (48 workers / 5 per event × 30s).
                # 5s floor still prevents thrashing while allowing 50s ramp-up.
                pool.spawn_delay = max(avg_t * 2, 5.0)

            # Effective ceiling: RSS budget limit capped at max_workers.
            # Capping prevents the grow branch from looping endlessly when the RSS budget
            # allows more workers than the physical core count.
            effective_max = min(cpu_budget_limit.get(pool, pool.max_workers), pool.max_workers)

            if pressure:
                pool.backoff()
            elif pool.current_concurrency_limit < effective_max:
                # Unified grow path for both post-backoff recovery and normal budget growth.
                # Grow slowly so EWMA has time to absorb each new worker before the next
                # spawns — prevents overshoot cycles where each spawn raises RSS, which
                # lowers effective_max, which immediately caps again.
                _grow_delay = min(pool.spawn_delay, self.resource_manager.limits.recovery_period_s) * self.resource_manager.limits.grow_delay_factor
                if time() - pool._last_recovery_time >= _grow_delay:
                    pool.current_concurrency_limit += 1
                    pool._last_recovery_time = time()
                    logger.info(
                        f"{pool.name}: {'recovering' if pool._in_backoff else 'growing'} "
                        f"to {pool.current_concurrency_limit}/{effective_max}"
                    )
                    if pool._in_backoff and pool.current_concurrency_limit >= effective_max:
                        pool._in_backoff = False
            elif pool._in_backoff:
                pool._in_backoff = False
                logger.info(f"{pool.name}: backoff recovery complete")

            # Proactive shedding: clamp to budget ceiling even outside backoff.
            # Log only when the budget ceiling itself changes (not on every EWMA tick
            # that repeats the same cap — EWMA converges in ~16 steps × 5 s = 80 s,
            # generating 16 distinct budget values, not 41 identical cap events).
            if pool.current_concurrency_limit > effective_max:
                prev_lim = pool.current_concurrency_limit
                pool.current_concurrency_limit = effective_max
                if self._last_cap_limit.get(pool) != effective_max:
                    logger.info(
                        f"{pool.name}: concurrency capped {prev_lim} → {effective_max} by RSS budget"
                    )
                    self._last_cap_limit[pool] = effective_max

            pool.scale()  # always: cleanup dead workers + reconcile alive → limit

    def _feeder_loop(self):
        """Background thread: continuously top up all pool queues."""
        while self.running:
            try:
                self._feed_pools()
            except Exception:
                logger.exception("Feeder loop error")
            sleep(0.02)

    def _log_status(self):
        """Log current resource and worker status."""
        active = self.running_jobs_per_device(None)
        worker_info = " | ".join(
            f"{pool.name}: {pool.alive_workers}w, q={pool.backlog}"
            for pool in self.worker_pools.values()
        )
        logger.info(
            f"{self.resource_manager.cpu_manager.get_status()} | "
            f"{active} active jobs | workers: {worker_info}"
        )
        if self.resource_manager.gpu_manager:
            logger.info(self.resource_manager.gpu_manager.get_status())

    def _scheduler_loop(self):
        """Main scheduler loop — collect results and detect lost jobs.

        Dispatching is handled by the feeder thread (_feeder_loop).
        This loop is purely reactive: it drains result queues, updates AIMD,
        and retries orphaned jobs.
        """
        logger.info(f"Scheduler loop started (total expected: {self.total_expected})")

        try:
            while self.running:
                for pool in self.worker_pools.values():
                    self._collect_results(pool)

                # Detect and retry jobs whose workers died without returning a result
                self._check_lost_jobs()

                for pool in self.worker_pools.values():
                    if pool._sigkill_detected:
                        pool._sigkill_detected = False
                        self.resource_manager.handle_oom(pool.device)
                        if pool.backoff():
                            logger.critical(f"System OOM kill detected on {pool.name} — shedding workers")

                if self.resource_manager.is_memory_pressure_critical():
                    avail = self.resource_manager.cpu_manager.get_system_usage()['memory_available_gb']
                    if any(pool.backoff() for pool in self.worker_pools.values()):
                        logger.critical(f"Memory pressure critical ({avail:.1f} GB available) — shedding workers")

                # Periodic status log (every 30 s)
                _now = time()
                if _now - self._last_status_time >= 30.0:
                    self._log_status()
                    self._last_status_time = _now

                finished = self._completed_count + len(self.permanently_failed)
                # In DB mode the node only runs a subset of jobs, so finished may never
                # reach total_expected. Exit when all claimed jobs are done instead.
                with self._in_flight_lock:
                    nothing_in_flight = not self.in_flight
                if self.submission_complete and (finished >= self.total_expected or nothing_in_flight):
                    logger.info("Scheduler loop exit condition met")
                    break

                sleep(self.scheduler_loop_delay)

        except Exception as e:
            logger.exception(f"Scheduler loop error: {e}")
        finally:
            logger.info("Scheduler loop exiting - draining remaining results")
            drain_deadline = time() + 30
            while self.in_flight and time() < drain_deadline:
                for pool in self.worker_pools.values():
                    self._collect_results(pool, timeout=1.0)
                self._check_lost_jobs()
                sleep(0.1)

            if self.in_flight:
                with self._in_flight_lock:
                    stuck = list(self.in_flight.keys())
                logger.warning(
                    f"Drain timeout: {len(stuck)} jobs still in flight: {stuck[:10]}"
                )
                for key in stuck:
                    with self._in_flight_lock:
                        self.in_flight.pop(key, None)
                    self._retry_job(key, 'drain timeout')
    
class _GridSearchProgress:
    """Progress bar for grid search — wraps tqdm with multi-node-aware display.

    n/total and elapsed are global. ETA, avg, and j/s are per-node:
    - ETA: how long this node alone would take to finish the remaining global work
    - avg: this node's average rate since start (node_done / elapsed)
    - j/s: this node's current rate (60s sliding window)
    ETA is computed manually so that n/total can stay global while the rate stays local.
    """

    _BAR_FMT = '{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}, {postfix}]'

    def __init__(self, total: int, initial: int):
        self._total = total
        self._rate_window: list[tuple[float, int]] = []
        self._pbar = tqdm(total=total, initial=initial, desc="Grid Search Progress",
                          unit="jobs", bar_format=self._BAR_FMT)
        self._redirect = logging_redirect_tqdm()
        self._stack = contextlib.ExitStack()

    def __enter__(self):
        self._stack.enter_context(self._redirect)
        self._stack.enter_context(self._pbar)
        return self

    def __exit__(self, *args):
        return self._stack.__exit__(*args)

    def update(self, global_done: int, node_done: int, elapsed_s: float) -> None:
        now = time()
        self._pbar.n = global_done

        self._rate_window.append((now, node_done))
        self._rate_window = [(t, n) for t, n in self._rate_window if now - t <= 60.0]
        if len(self._rate_window) >= 2:
            dt = self._rate_window[-1][0] - self._rate_window[0][0]
            dn = self._rate_window[-1][1] - self._rate_window[0][1]
            cur_rate = dn / dt if dt > 0 else 0.0
        else:
            cur_rate = 0.0

        avg_rate = node_done / elapsed_s
        complete = global_done >= self._total
        if complete:
            self._pbar.set_description("Compacting")
            self._pbar.set_postfix({'this_node': node_done, 'j/s': f'{cur_rate:.2f}'})
        else:
            remaining = self._total - global_done
            eta_s = remaining / avg_rate if avg_rate > 0 else float('inf')
            eta_str = tqdm.format_interval(int(eta_s)) if eta_s != float('inf') else '?'
            self._pbar.set_postfix({'eta': eta_str, 'this_node': node_done, 'j/s': f'{cur_rate:.2f}', 'avg': f'{avg_rate:.2f}'})
        self._pbar.refresh()


def _reconstruct_state(
    output_path: Path,
    total_configs: int,
    samples_per_config: int,
) -> set:
    """Scan data/*.parquet to collect completed (i,j) pairs, then build state.db atomically.

    state.db and data.parquet live inside data/ permanently. After compaction,
    batch parquets and lock/tmp files are removed but state.db and data.parquet
    are kept so check_grid.py can inspect completed runs.

    Writes to data/state.db.tmp then renames so any reader seeing data/state.db
    can trust it is complete.
    """
    data_dir = output_path / RUN.data_dir
    done_pairs: set = set()

    if data_dir.is_dir():
        for pq_file in sorted(data_dir.glob('*.parquet')):
            try:
                t = pq.read_table(pq_file, columns=['i', 'j'])
                for i, j in zip(t.column('i').to_pylist(), t.column('j').to_pylist()):
                    done_pairs.add((i, j))
            except Exception as e:
                logger.warning(f"Skipping unreadable batch file during reconstruction: {pq_file.name}: {e}")

    tmp_path = data_dir / (RUN.state_db + '.tmp')
    state_path = data_dir / RUN.state_db

    for suffix in ('', '-journal', '-wal', '-shm'):
        (data_dir / (tmp_path.name + suffix)).unlink(missing_ok=True)

    db = GridSearchDB(tmp_path)
    db.init_jobs(total_configs, samples_per_config)
    if done_pairs:
        db.mark_done_batch(list(done_pairs))
    db.close()

    try:
        os.replace(str(tmp_path), str(state_path))
    except FileNotFoundError:
        # Two nodes both won the O_CREAT|O_EXCL race on NFS (non-atomic).
        # The other node already renamed state.db.tmp → state.db; our tmp is gone.
        # state.db is consistent — just continue.
        if not state_path.exists():
            raise  # genuinely missing, not a race win by another node
    return done_pairs


def _ensure_state_db(
    output_path: Path,
    total_configs: int,
    samples_per_config: int,
    node_id: str = '',
) -> 'GridSearchDB':
    """Thundering-herd protocol: exactly one node builds state.db, others wait.

    If state.db already exists, opens it immediately (fast path) and resets any
    claimed jobs from dead nodes (heartbeat-based crash recovery).  Otherwise,
    races to atomically create BUILDING_LOCK.  The winner runs _reconstruct_state
    and removes the lock; losers wait, watching for the lock to disappear and
    state.db to appear.

    If the lock disappears without state.db appearing, the winner must have crashed
    mid-reconstruction — retry the whole protocol.
    """
    data_dir = output_path / RUN.data_dir
    state_path = data_dir / RUN.state_db
    building_lock_path = data_dir / RUN.building_lock

    if state_path.exists():
        db = GridSearchDB.open(state_path, total_configs, samples_per_config, node_id=node_id)
        n_stale = db.reset_stale_claimed()
        if n_stale:
            logger.warning(f"Reset {n_stale} stale claimed jobs from dead nodes")
        _startup_counts = db.counts()
        logger.info(
            f"Loaded state.db: {_startup_counts.get('done', 0)}/{total_configs * samples_per_config} done, "
            f"{_startup_counts.get('claimed', 0)} claimed (in-flight from previous run)"
        )
        return db

    # Race to win reconstruction: O_CREAT|O_EXCL is atomic on POSIX local filesystems.
    # On NFS it is not guaranteed, but the worst case is two nodes both win and both
    # call _reconstruct_state — they both write to state.db.tmp then os.replace
    # (atomic rename), so the last writer wins and state.db is still consistent.
    try:
        fd = os.open(str(building_lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
        winner = True
    except FileExistsError:
        winner = False

    if winner:
        try:
            logger.info(f"Won state reconstruction lock — scanning {RUN.data_dir}/ for completed jobs")
            done_pairs = _reconstruct_state(output_path, total_configs, samples_per_config)
            logger.info(f"State reconstruction complete: {len(done_pairs)} previously completed jobs recovered")
        finally:
            building_lock_path.unlink(missing_ok=True)
    else:
        logger.warning(
            f"State reconstruction in progress — another node holds {RUN.building_lock}. "
            f"Waiting for state.db to appear. "
            f"If this message persists for >5 minutes (possible crashed node), "
            f"delete {building_lock_path} and restart."
        )
        deadline = time() + 300.0
        while not state_path.exists():
            if time() > deadline:
                raise RuntimeError(
                    f"Timed out after 300s waiting for {RUN.state_db} to appear. "
                    f"If {RUN.building_lock} is stale (winner node crashed mid-reconstruction), "
                    f"delete {building_lock_path} and restart."
                )
            if not building_lock_path.exists() and not state_path.exists():
                logger.warning(
                    f"{RUN.building_lock} disappeared without {RUN.state_db} appearing — "
                    f"winner may have crashed; retrying reconstruction"
                )
                return _ensure_state_db(output_path, total_configs, samples_per_config, node_id)
            sleep(2.0)

    db = GridSearchDB.open(state_path, total_configs, samples_per_config, node_id=node_id)
    done = db.counts().get('done', 0)
    logger.info(f"State.db ready: {done}/{total_configs * samples_per_config} done")
    return db


def _read_batch_files(data_dir: Path) -> tuple[list, list[Path]]:
    """Read all Parquet files in data_dir for compaction. Returns (tables, pq_files).

    Includes data.parquet (previous partial compaction) so results whose original
    batch files were cleaned up after an earlier successful compaction are not lost.
    The regression guard in _dedup_and_write prevents overwriting data.parquet with
    fewer rows if a stale-cache-triggered re-compaction sees fewer batch files.
    """
    pq_files = sorted(data_dir.glob('*.parquet'))
    tables = []
    for f in pq_files:
        try:
            tables.append(pq.read_table(f))
        except Exception as e:
            logger.warning(f"Skipping unreadable batch file during compaction: {f.name}: {e}")
    return tables, pq_files


def _dedup_and_write(df, data_dir: Path, n_files: int) -> int:
    """Dedup on (i,j), write compacted parquet atomically. Returns unique row count.

    Regression guard: never overwrites an existing data.parquet with fewer rows.
    This protects against a stale-cache-triggered re-compaction on NFS writing a
    partial result over an already-complete file.
    """
    import pandas as pd
    before = len(df)
    df = df.drop_duplicates(subset=['i', 'j'], keep='last').reset_index(drop=True)
    if len(df) < before:
        logger.info(f"Compaction: removed {before - len(df)} duplicate rows")
    existing = data_dir / RUN.compacted_file
    if existing.exists():
        try:
            existing_n = pq.read_metadata(str(existing)).num_rows
            if len(df) < existing_n:
                logger.info(
                    f"Skipping write — new result ({len(df)} rows) < existing ({existing_n} rows); "
                    f"keeping the more complete version"
                )
                return existing_n
        except Exception:
            pass  # unreadable existing file — proceed with overwrite
    result_table = pa.Table.from_pandas(df, preserve_index=False)
    tmp = data_dir / (RUN.compacted_file + '.tmp')
    pq.write_table(result_table, tmp)
    os.replace(str(tmp), str(data_dir / RUN.compacted_file))
    logger.info(f"Compacted {n_files} batch files ({len(df)} unique results) → data/data.parquet")
    return len(df)


def _compact_results(data_dir: Path, compact_transform=None) -> None:
    """Merge all batch Parquet files in data/ into data/data.parquet.

    Skips any existing data.parquet (previous compaction) as input — it will be
    overwritten as the output. Deduplicates on (i,j) keeping the last occurrence,
    then writes atomically via a .tmp file rename.

    compact_transform: optional callable (pd.DataFrame) -> pd.DataFrame applied
    to the merged DataFrame before writing. Use for one-off data migrations.
    """
    import pandas as pd

    if not data_dir.is_dir():
        logger.warning("data/ directory not found — nothing to compact")
        return

    tables, pq_files = _read_batch_files(data_dir)
    if not pq_files:
        logger.warning("No Parquet files found in data/ — data.parquet not written")
        return
    compacted = data_dir / RUN.compacted_file
    if pq_files == [compacted]:
        # Only data.parquet exists — nothing new to merge; return its current row count.
        try:
            return pq.read_metadata(str(compacted)).num_rows
        except Exception:
            pass  # fall through to full compaction if unreadable
    if not tables:
        logger.warning("All batch files were unreadable — data.parquet not written")
        return

    df = pa.concat_tables(tables).to_pandas()
    if compact_transform is not None:
        df = compact_transform(df)
    return _dedup_and_write(df, data_dir, len(pq_files))


def generic_parallel_grid_search(
    # Core parameters
    job_factory: Callable[..., JobInterface],
    total_configs: int,
    samples_per_config: int,
    output_path: Path,
    save_config: Callable[[Path], None],
    process_results: Callable[[List, List, Path], Any],
    # Resource parameters
    cpu_memory_per_job_gb: float = 1,
    cpu_cores_per_job: int = 1,
    cpu_max_workers: int = None,
    history_write_thresh: int = 1000,
    # Routing parameters
    exploration_rate: float = 0.1,
    # Retry parameters
    max_retries: int = 3,
    compact: bool = True,
    compact_transform=None,
    limits: ResourceLimits | None = None,
) -> Tuple[List[Dict], Dict]:
    """Generic parallel grid search.

    Results accumulate as batch Parquet files in output_path/data/ during the run.
    When all jobs globally complete, exactly one node compacts them into data.parquet
    via an NFS-safe SQLite lock, then writes GRID_SEARCH_COMPLETED.flag.

    State (pending/done) is tracked in output_path/state.db — an ephemeral SQLite
    file rebuilt from data/*.parquet on startup if missing.  On restart after a
    crash, _ensure_state_db re-scans data/ so only jobs without saved results re-run.

    Signature of process_results callback: (entries, marks, batch_file: Path) → any
    """

    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    flag_path = output_path / RUN.completed_flag
    if flag_path.exists():
        logger.info(f"{RUN.completed_flag} exists — grid search already complete, exiting")
        return

    data_dir = output_path / RUN.data_dir
    data_dir.mkdir(exist_ok=True)

    import socket as _socket
    _node_name = f'{_socket.gethostname()}_{os.getpid()}_{int(time())}'

    params_file = output_path / 'parameters.yaml'
    if params_file.exists():
        with tempfile.TemporaryDirectory() as _tmpdir:
            save_config(Path(_tmpdir))
            new_content = (Path(_tmpdir) / 'parameters.yaml').read_text()
        existing_content = params_file.read_text()
        if new_content.strip() != existing_content.strip():
            raise RuntimeError(
                f"\nConfig mismatch — refusing to resume grid search at:\n"
                f"  {output_path}\n\n"
                f"The parameters.yaml on disk differs from the current config.\n"
                f"This usually means a YAML setting was changed after the run started.\n"
                f"To start fresh: delete {output_path} or point to a new output path."
            )
    else:
        save_config(output_path)

    db = _ensure_state_db(output_path, total_configs, samples_per_config, node_id=_node_name)

    resource_manager = ComputeJobResourceManager(
        cpu_memory_per_job_gb=cpu_memory_per_job_gb,
        cpu_cores_per_job=cpu_cores_per_job,
        cpu_max_workers=cpu_max_workers,
        exploration_rate=exploration_rate,
        limits=limits,
    )

    history = []
    shutdown = threading.Event()

    def _on_signal(signum, frame):
        logger.info(f"Received signal {signal.Signals(signum).name}, shutting down gracefully")
        shutdown.set()

    _batch_seq = 0

    def write_history(flush=False, thresh=history_write_thresh):
        nonlocal _batch_seq
        if not flush and len(history) < thresh:
            return
        with scheduler._history_lock:
            n = len(history)
            entries = history[:n]
            del history[:n]
            # Consume exactly n marks — always paired with entries under the lock.
            marks = scheduler._pending_marks[:n]
            del scheduler._pending_marks[:n]

        if entries:
            _batch_seq += 1
            batch_file = data_dir / f'{_node_name}_{_batch_seq:06d}.parquet'
            process_results(entries, marks, batch_file)
            db.mark_done_batch(marks)  # transition claimed→done now that results are on disk
            logger.info(f"Wrote batch {_batch_seq}: {len(entries)} results [{scheduler._completed_count}/{total_jobs}]  e.g. {entries[0]}")

    # Heartbeat daemon: stamps this node alive every HEARTBEAT_INTERVAL_S seconds.
    # Other nodes use the heartbeats table to detect crashes and reclaim stale jobs.
    _HEARTBEAT_INTERVAL_S = GridSearchDB.HEARTBEAT_TIMEOUT_S // 5  # 60 s default
    _hb_stop = threading.Event()
    def _heartbeat_loop():
        db.update_heartbeat(_node_name)
        while not _hb_stop.wait(_HEARTBEAT_INTERVAL_S):
            try:
                db.update_heartbeat(_node_name)
            except Exception as e:
                logger.warning(f"Heartbeat write failed: {e}")
    _hb_thread = threading.Thread(target=_heartbeat_loop, daemon=True, name='heartbeat')
    _hb_thread.start()

    prev_sigterm = signal.signal(signal.SIGTERM, _on_signal)
    prev_sigint = signal.signal(signal.SIGINT, _on_signal)
    try:
        # Dataset lock: node-local, keeps long dataset builds off state.db.
        # 10-minute timeout: covers 64 workers × ~10s/load without false timeouts.
        dataset_lock = SQLiteLock(Path(tempfile.gettempdir()) / 'dataset.lock.db',
                                  timeout_ms=600_000)
        job_generator = JobGenerator(
            job_factory=job_factory,
            total_configs=total_configs,
            samples_per_config=samples_per_config,
            db=db,
            locks={DATASET_LOCK_KEY: dataset_lock},
        )
        with ResourceAwareScheduler(resource_manager=resource_manager, history=history, job_generator=job_generator, max_retries=max_retries, db=db) as scheduler:
            start_time = time()
            total_jobs = total_configs * samples_per_config
            _initial = db.counts()
            _initial_done = _initial.get('done', 0)
            with _GridSearchProgress(total=total_jobs, initial=_initial_done) as pbar:
                _last_counts_t = 0.0
                _last_stale_reset_t = 0.0
                overall = _initial
                while not shutdown.is_set():
                    now_t = time()
                    if now_t - _last_counts_t >= 10.0:
                        overall = db.counts()
                        _last_counts_t = now_t
                    if now_t - _last_stale_reset_t >= GridSearchDB.HEARTBEAT_TIMEOUT_S:
                        n_stale = db.reset_stale_claimed()
                        if n_stale:
                            logger.warning(f"Reclaimed {n_stale} jobs from dead nodes")
                        _last_stale_reset_t = now_t
                    global_done = overall.get('done', 0)
                    node_done = scheduler.completed_count + len(scheduler.permanently_failed)
                    elapsed_s = max(1.0, time() - start_time)
                    pbar.update(global_done, node_done, elapsed_s)
                    thresh = 1 if scheduler._completed_count <= 1 else history_write_thresh
                    write_history(thresh=thresh)
                    scheduler_finished = not scheduler.scheduler_thread.is_alive()
                    if scheduler_finished:
                        break
                    sleep(0.5)

            if shutdown.is_set():
                logger.info(f"Shutdown requested — completed {scheduler.completed_count}/{total_jobs} jobs, {len(scheduler.permanently_failed)} failed")

        # Scheduler thread is joined at this point — flush remaining results.
        write_history(flush=True)

        # Compaction: exactly one node merges data/*.parquet → data.parquet.
        # Claim election uses db.try_claim_compaction() (BEGIN IMMEDIATE on state.db,
        # milliseconds) so only one node does the heavy file I/O.  All other nodes
        # skip immediately, keeping mark_done_batch / update_heartbeat unblocked.
        _global_counts = db.counts()
        # Use only status=2 (done = result written to parquet) as the trigger.
        # Including status=1 (claimed = in flight) caused premature compaction: a node
        # that finishes early sees claimed+done==total even though other nodes haven't
        # flushed yet, compacts with a partial row count, and exits — leaving a gap.
        _n_done   = _global_counts.get('done', 0)
        _n_pending = _global_counts.get('pending', 0) + _global_counts.get('claimed', 0)
        if not compact:
            logger.info("Compaction disabled — results remain as batch files in data/")
        elif _n_pending > 0:
            logger.info(
                f"Skipping compaction: {_n_done}/{total_jobs} results flushed, "
                f"{_n_pending} jobs still in flight — another node will compact when complete"
            )
        else:
            if flag_path.exists():
                logger.info(f"{RUN.completed_flag} already written — another node handled compaction")
            elif not db.try_claim_compaction():
                # Another node atomically inserted the sentinel first — let it handle compaction.
                logger.info("Another node claimed compaction — skipping")
            else:
                # This node won the claim.  Heavy I/O runs outside any lock so other nodes'
                # mark_done_batch / update_heartbeat calls are never blocked.
                logger.info(f"All {total_jobs} jobs done — compacting results")
                n_written = _compact_results(data_dir, compact_transform=compact_transform)

                def _finish(flag_msg):
                    if not flag_path.exists():
                        flag_path.touch()
                        logger.info(flag_msg)
                    if flag_path.exists() and (data_dir / RUN.compacted_file).exists():
                        _keep = {RUN.state_db, RUN.compacted_file}
                        for f in data_dir.iterdir():
                            if f.is_file() and f.name not in _keep:
                                f.unlink(missing_ok=True)
                        logger.info(f"Cleaned up batch files from {RUN.data_dir}/ (kept {RUN.state_db} + {RUN.compacted_file})")

                if n_written is None:
                    logger.warning(
                        "Compaction found no batch files — results not yet flushed by another node. "
                        "Another node will compact when its results are written."
                    )
                elif n_written >= total_jobs:
                    _finish(f"Wrote {RUN.completed_flag}")
                else:
                    n_gap = total_jobs - n_written
                    if _n_pending == 0:
                        # No in-flight jobs left — gap is permanently failed jobs
                        # (marked done in state.db but produced no parquet result).
                        logger.warning(
                            f"{n_gap} permanently failed jobs have no parquet result "
                            f"({n_written}/{total_jobs} rows written)"
                        )
                        _finish(f"Wrote {RUN.completed_flag} (with {n_gap} failed jobs)")
                    else:
                        logger.info(
                            f"Compacted {n_written}/{total_jobs} rows — flag deferred until "
                            f"remaining batch files are flushed by in-flight nodes"
                        )
                db.release_compaction_claim()  # remove sentinel so future restarts can compact
            db.close()

        elapsed_time = time() - start_time
        logger.info(f"Grid search {'interrupted' if shutdown.is_set() else 'completed'} in {elapsed_time//3600:.1f}hrs")

    except Exception as e:
        logger.exception(f"Unexpected error in grid search: {e}")
        raise
    finally:
        _hb_stop.set()              # signal heartbeat loop to exit
        _hb_thread.join(timeout=5)  # wait so it finishes any in-flight update_heartbeat before db.close()
        signal.signal(signal.SIGTERM, prev_sigterm)
        signal.signal(signal.SIGINT, prev_sigint)