import collections
from pathlib import Path
from time import sleep, time
import queue
import random
import numpy as np  # Must be imported before torch to avoid MKL/OpenMP threading conflicts in multiprocessing
import torch.multiprocessing as tmp
from torch import set_num_threads, set_num_interop_threads
import psutil
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
import threading
from typing import Tuple, Callable, List, Dict, Any
from project.parallel_grid_search.code.parallel_utils import GenericJobGenerator, ComputeJobResourceManager, JobInterface # TODO add type hints to these
import errno
import os
from os import waitpid, WNOHANG, environ
import torch
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
    environ['TORCHINDUCTOR_CACHE_DIR'] = str(tmp_dir / 'torch_cache')
_setup_node_tmp()

# --- WORKAROUND: Disable resource_tracker for semaphores ---
# All stdlib mp.Queue and SimpleQueue implementations use SemLock (named POSIX
# semaphores) internally.  The resource_tracker is meant to clean these up when
# processes die, but it cannot distinguish "I inherited this" from "I created
# this" — so it unlinks semaphores that are still in use by the parent's queues,
# causing FileNotFoundError on the next worker spawn.
# Fix: never register semaphores with the tracker.  The parent cleans up queues
# via queue.close() + join_thread(); Slurm reaps the cgroup on job exit.
# TODO: Remove when CPython fixes resource_tracker lifetime scoping.
#       Track: https://github.com/python/cpython/issues/82300
from multiprocessing import resource_tracker as _rt
_rt_register = _rt.register
_rt_unregister = _rt.unregister

def _register_except_semaphore(name, rtype):
    if rtype == "semaphore":
        return
    _rt_register(name, rtype)

def _unregister_except_semaphore(name, rtype):
    if rtype == "semaphore":
        return
    _rt_unregister(name, rtype)

_rt.register = _register_except_semaphore
_rt.unregister = _unregister_except_semaphore

# --- WORKAROUND: Suppress spurious FileNotFoundError from SemLock finalizers ---
# With spawn mode, each child unpickles a new SemLock wrapping the same POSIX
# semaphore name and registers its own _cleanup finalizer. The child unlinks
# the semaphore on exit; when the parent interpreter shuts down, its own
# finalizer tries to unlink the already-gone semaphore → FileNotFoundError.
# This is safe: if the semaphore was already removed, there is nothing to clean up.
# TODO: Remove when CPython fixes SemLock finalizer ownership scoping.
#       Track: https://github.com/python/cpython/issues/82300
from multiprocessing import synchronize as _mp_sync
_orig_sem_cleanup = _mp_sync.SemLock._cleanup  # staticmethod → plain function

@staticmethod
def _safe_sem_cleanup(name):
    try:
        _orig_sem_cleanup(name)
    except FileNotFoundError:
        pass

_mp_sync.SemLock._cleanup = _safe_sem_cleanup

# --- TORCH-AWARE START METHOD ---
if tmp.get_start_method(allow_none=True) is None:
    tmp.set_start_method('spawn')

import logging
logger = logging.getLogger(__name__)


def _cuda_health_check(device):
    """Quick CUDA health check — returns True if device is usable, False otherwise."""
    try:
        test = torch.zeros(1, device=device)
        del test
        return True
    except Exception as e:
        logger.error(f"CUDA health check FAILED on {device}: {type(e).__name__}: {e}", exc_info=True)
        return False


def worker_function(job_queue, result_queue, cores_per_job, priority, device,
                    idle_timeout: float = 120.0):
    """Worker function - no shared state, just queues.

    Workers live until the queue is empty for idle_timeout seconds (starvation exit)
    or they receive a None sentinel (soft kill / scale-down).
    idle_timeout is the laziness knob: longer = more resistant to brief W dips.
    """
    pid = os.getpid()
    set_num_interop_threads(1)
    set_num_threads(cores_per_job)
    p = psutil.Process()

    if device.type == 'cuda':
        try:
            torch.cuda.set_device(device.index)
        except Exception as e:
            logger.error(f"Worker PID {pid}: torch.cuda.set_device({device.index}) FAILED: {e}", exc_info=True)
            return
        if not _cuda_health_check(device):
            logger.error(f"Worker PID {pid}: CUDA {device.index} unusable at startup, exiting")
            return
        logger.debug(f"Worker PID {pid}: CUDA {device.index} health check passed")

    try:
        p.nice(priority)
    except (psutil.AccessDenied, PermissionError):
        pass

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
        result_queue.put({'status': 'started', 'key': (job.i, job.j), 'pickup_time': start_time})
        try:
            result = job.run(device)
            job.locks = {}  # clear before pickling back through result_queue
            result.update({
                'job': job,
                'device': device,
                'worker_pid': pid,
                'start_time': start_time,
                'stop_time': time(),
            })
        except Exception as e:
            logger.error(f"Worker PID {pid}: UNHANDLED exception in job {job.i}-{job.j} on {device}: {e}", exc_info=True)
            if device.type == 'cuda':
                try:
                    mem_summary = torch.cuda.memory_summary(device, abbreviated=True)
                    logger.error(f"Worker PID {pid}: CUDA memory state on {device}:\n{mem_summary}")
                except Exception:
                    pass
                cuda_ok = _cuda_health_check(device)
                logger.error(f"Worker PID {pid}: CUDA health after crash: {'OK' if cuda_ok else 'FAILED'}")
            job.locks = {}
            result = {
                'status': 'error',
                'error_type': 'worker_crash',
                'error': f'Unhandled {type(e).__name__}: {e}',
                'job': job,
                'device': device,
                'worker_pid': pid,
                'start_time': start_time,
                'stop_time': time(),
            }

        try:
            result_queue.put(result)
        except Exception as e:
            logger.error(f"Result serialization failed for job {job.i}-{job.j}: {e}")
            try:
                result_queue.put({
                    'status': 'error',
                    'error_type': 'serialization',
                    'error': f'Result serialization failed: {type(e).__name__}: {e}',
                    'job': job,
                    'device': device,
                    'worker_pid': pid,
                    'start_time': start_time,
                    'stop_time': time(),
                })
            except Exception as e2:
                logger.critical(
                    f"Even fallback result failed for job {job.i}-{job.j}: {e2}"
                )

        if hasattr(result['job'], 'cleanup'):
            result['job'].cleanup()

        logger.debug(f"Worker completed job {result['job'].i}-{result['job'].j} on {device}")
    logger.debug(f"Worker PID {pid}: exiting on {device}")


def worker_function_cpu(*args, **kwargs):
    environ['CUDA_VISIBLE_DEVICES'] = ''  # hide GPU, but torch.compile still works
    return worker_function(*args, **kwargs)

def zombie_reaping():
    while True:
        try:
            pid, _ = waitpid(-1, WNOHANG)
            if pid == 0:
                break
        except OSError as e:
            if e.errno == errno.ECHILD:
                break

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
                 can_spawn_fn=None, idle_timeout: float = 120.0,
                 spawn_delay: float = 3.0, max_scale: int = 1):
        self.device = torch.device(device)
        self.max_workers = max_workers
        self.cores_per_job = cores_per_job
        self.process_priority = process_priority
        self.can_spawn_fn = can_spawn_fn or (lambda pending=0: True)
        self.idle_timeout = idle_timeout
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
        self._concurrency_limit = 1 if self.device.type == 'cuda' else max_workers

        self.name = device + '_pool'
        logger.info(f"{self.name} initialized (max_workers={max_workers})")

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
        target = worker_function if self.device.type == 'cuda' else worker_function_cpu
        p = tmp.Process(
            target=target,
            args=(self.worker_job_queue, self.result_queue, self.cores_per_job,
                  self.process_priority, self.device, self.idle_timeout),
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
            logger.info(f"{self.name}: Initiating shutdown")
            
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
                
            logger.info(f"{self.name}: shutdown complete")
    
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


class ResourceAwareScheduler:
    """Simplified scheduler with centralized resource management"""


    def __init__(self, resource_manager: ComputeJobResourceManager, history: list, job_generator: GenericJobGenerator, scheduler_loop_delay: int = .1, max_retries: int = 3):
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
        _physical_cores = self.resource_manager.cpu_manager.physical_cpus
        _logical_available = self.resource_manager.max_concurrent
        _effective_cores = max(1, int(min(_physical_cores, _logical_available) * 0.9))
        cpu_pool = LazyWorkerPool(
            max_workers=_effective_cores,
            cores_per_job=self.resource_manager.cores_per_job,
            process_priority=15, can_spawn_fn=_make_can_spawn(torch.device('cpu')),
            max_scale=max(1, int(_effective_cores * 0.1)),
        )
        self.worker_pools[cpu_pool.device] = cpu_pool
        if self.resource_manager.use_gpu:
            for gpu_id in self.resource_manager.gpu_manager.available_gpus:
                gpu_device = torch.device(f'cuda:{gpu_id}')
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
        self.in_flight = {}                # (i,j) -> {'time': float, 'device': torch.device}

        self._completed_count = 0          # jobs that completed successfully
        self.permanently_failed = set()    # (i,j) pairs that exceeded max retries
        self.retry_counts = {}             # (i,j) -> int
        self.max_retries = max_retries
        self.total_expected = len(job_generator)

        # Dispatch cap: upper safety ceiling on total jobs in flight (queued + running).
        # in_flight includes backlog sitting in pool queues, not just active workers.
        # Scale by pool count so each pool can hold a full max_concurrent-deep queue.
        # Worker process count is still gated independently by can_spawn_fn / max_workers.
        self.max_in_flight = self.resource_manager.max_concurrent * len(self.worker_pools)

        self._last_status_time = 0.0

        self._start_job_submissions(job_generator)

    def running_jobs_per_device(self, device):
        """Jobs actively being processed: dispatched to a pool minus jobs still in its queue."""
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
        self.in_flight.pop(key, None)

        if result['status'] == 'completed':
            exec_time  = result['stop_time'] - result['start_time']
            self.resource_manager.bee_router.record_completion(device, exec_time)
            if 'history' in result:
                self.history.append(result['history'])
            self._completed_count += 1
            logger.info(f"Completed job {job.i}-{job.j} on device {device}")

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
        logger.info(f"{pool.name} - {pool.device} recovered from OOM")

    def _retry_job(self, key, reason):
        """Re-create and re-queue a failed/lost job, or mark permanently failed."""
        i, j = key
        count = self.retry_counts.get(key, 0) + 1
        self.retry_counts[key] = count

        if count > self.max_retries:
            logger.error(f"Job {i}-{j} permanently failed after {self.max_retries} retries: {reason}")
            self.permanently_failed.add(key)
            return

        logger.warning(f"Retrying job {i}-{j} (attempt {count}/{self.max_retries}): {reason}")
        logging.getLogger().setLevel(logging.DEBUG)
        job = self.job_generator.job_factory(
            i=i, j=j,
            total_configs=self.job_generator.total_configs,
            total_samples=self.job_generator.samples_per_config,
            locks=self.job_generator.locks,
        )
        self._retry_queue.append(job)

    def _check_lost_jobs(self):
        """Detect and retry jobs whose worker pool died without returning a result."""
        now = time()

        for device, pool in self.worker_pools.items():
            if pool.alive_workers == 0 and pool.backlog == 0:
                # Drain the result queue before declaring jobs lost.
                # A worker may have completed its job and pushed the result before
                # dying. Collecting those results now removes them from in_flight,
                # preventing duplicate dispatch of jobs that already finished.
                while True:
                    result = pool.get_result(timeout=0.0)
                    if result is None:
                        break
                    if result.get('status') == 'started':
                        k = result['key']
                        if k in self.in_flight:
                            self.in_flight[k]['pickup_time'] = result['pickup_time']
                    else:
                        self._handle_job_completion(result)

                device_jobs = [(k, v) for k, v in list(self.in_flight.items()) if v['device'] == device]
                for key, info in device_jobs:
                    self.in_flight.pop(key)
                    age = now - info['queued_time']
                    logger.error(
                        f"Job {key[0]}-{key[1]} LOST on {device}: pool has 0 alive workers "
                        f"(job age: {age:.0f}s)"
                    )
                    self._retry_job(key, f'device pool dead ({device})')
    
    def _feed_pools(self):
        """Kanban fill: top up every pool queue to 2 * current_concurrency_limit.

        Pools are shuffled so no device starves the others when
        scheduler_job_queue runs low. _try_scale_up (called after filling)
        always invokes _cleanup_dead_workers, keeping _alive_count accurate.
        """
        if not self.running:
            return

        pools = list(self.worker_pools.values())
        random.shuffle(pools)

        # Sync AIMD limits to GPU pools before spawning decisions
        for pool in pools:
            if pool.device.type == 'cuda':
                pool.current_concurrency_limit = (
                    self.resource_manager.bee_router._concurrency_limit.get(pool.device, 1)
                )

        global_budget = max(0, self.max_in_flight - len(self.in_flight))

        for pool in pools:
            if not pool.enabled:
                continue
            if pool.device.type == 'cuda':
                # GPU queue filling: only gate on GPU memory, not CPU load.
                # CPU load is the right gate for *worker spawning* (via can_spawn_fn),
                # but applying it here too starves GPU queues whenever CPU is saturated.
                if not self.resource_manager.gpu_manager._can_allocate_on_gpu(pool.device.index):
                    continue
            elif not self.resource_manager.try_allocate_for_device(pool.device):
                continue

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
                self.in_flight[(job.i, job.j)] = {
                    'queued_time': time(), 'pickup_time': None, 'device': pool.device
                }
                global_budget -= 1

        self._scale_pools()

    def _scale_pools(self):
        """Adjust each pool toward its W target, rate-limited by spawn_delay.

        Sets spawn_delay from avg job time (laziness adapts to workload speed)
        then delegates all scale logic — including dead-worker cleanup — to pool.scale().
        """
        for pool in self.worker_pools.values():
            avg_t = self.resource_manager.bee_router._last_time.get(pool.device, None)
            if avg_t is not None:
                pool.spawn_delay = avg_t
            pool.scale()

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
                zombie_reaping()

                for pool in self.worker_pools.values():
                    while True:
                        result = pool.get_result(timeout=0.0)
                        if result is None:
                            break
                        if result.get('status') == 'started':
                            key = result['key']
                            if key in self.in_flight:
                                self.in_flight[key]['pickup_time'] = result['pickup_time']
                        else:
                            self._handle_job_completion(result)

                # Detect and retry jobs whose workers died without returning a result
                self._check_lost_jobs()

                # Periodic status log (every 30 s)
                _now = time()
                if _now - self._last_status_time >= 30.0:
                    self._log_status()
                    self._last_status_time = _now

                finished = self._completed_count + len(self.permanently_failed)
                if self.submission_complete and finished >= self.total_expected:
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
                    result = pool.get_result(timeout=1.0)
                    if result is not None:
                        self._handle_job_completion(result)
                self._check_lost_jobs()
                zombie_reaping()
                sleep(0.1)

            if self.in_flight:
                logger.warning(
                    f"Drain timeout: {len(self.in_flight)} jobs still in flight: "
                    f"{list(self.in_flight.keys())[:10]}"
                )
                for key in list(self.in_flight.keys()):
                    self.in_flight.pop(key)
                    self._retry_job(key, 'drain timeout')
    
def generic_parallel_grid_search(
    # Core parameters
    job_factory: Callable,
    total_configs: int,
    samples_per_config: int,
    output_path: Path,
    save_config: Callable[[Path], None],
    process_results: Callable[[List[Dict], Path, bool], Any],
    cleanup: Callable = None,
    # Resource parameters
    gpu_memory_per_job_gb: float = 1,
    cpu_memory_per_job_gb: float = 1,
    cpu_cores_per_job: int = 1,
    history_write_thresh: int = 1000,
    # Routing parameters
    exploration_rate: float = 0.1,
    # Retry parameters
    max_retries: int = 3,
    ) -> Tuple[List[Dict], Dict]:
    """Generic parallel grid search that works with any job type."""
    
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=False)
    save_config(output_path)
    
    resource_manager = ComputeJobResourceManager(
        cpu_memory_per_job_gb=cpu_memory_per_job_gb,
        cpu_cores_per_job=cpu_cores_per_job,
        gpu_memory_per_job_gb=gpu_memory_per_job_gb,
        exploration_rate=exploration_rate,
    )

    history = []
    shutdown = threading.Event()

    def _on_signal(signum, frame):
        logger.info(f"Received signal {signal.Signals(signum).name}, shutting down gracefully")
        shutdown.set()

    def write_history(flush=False, thresh=history_write_thresh):
        if not flush and len(history) < thresh:
            return
        entries = history[:]
        del history[:]
        if entries:
            process_results(entries, output_path, flush)
            logger.info(f"Wrote {len(entries)} results to {output_path}")
        elif flush:
            process_results([], output_path, True)

    prev_sigterm = signal.signal(signal.SIGTERM, _on_signal)
    prev_sigint = signal.signal(signal.SIGINT, _on_signal)
    try:
        job_generator = GenericJobGenerator(
            job_factory=job_factory,
            total_configs=total_configs,
            samples_per_config=samples_per_config,
        )
        with ResourceAwareScheduler(resource_manager=resource_manager, history=history, job_generator=job_generator, max_retries=max_retries) as scheduler:
            start_time = time()
            with logging_redirect_tqdm():
                total_jobs = len(job_generator)
                with tqdm(total=total_jobs, desc="Grid Search Progress", unit="jobs") as pbar:
                    while pbar.n < total_jobs and not shutdown.is_set():
                        pbar.n = scheduler.completed_count + len(scheduler.permanently_failed)
                        pbar.refresh()
                        write_history()
                        sleep(0.5)

            if shutdown.is_set():
                logger.info(f"Shutdown requested — completed {scheduler.completed_count}/{total_jobs} jobs, {len(scheduler.permanently_failed)} failed")

            if cleanup is not None:
                cleanup()

        # After scheduler.stop() (via __exit__), scheduler thread is joined and
        # all results are guaranteed to be in history before the final flush.
        write_history(flush=True)
        elapsed_time = time() - start_time
        logger.info(f"Grid search {'interrupted' if shutdown.is_set() else 'completed'} in {elapsed_time//3600:.1f}hrs")

    except Exception as e:
        logger.exception(f"Unexpected error in grid search: {e}")
        raise
    finally:
        signal.signal(signal.SIGTERM, prev_sigterm)
        signal.signal(signal.SIGINT, prev_sigint)