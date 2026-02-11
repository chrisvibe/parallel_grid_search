from pathlib import Path
from time import sleep, time
import queue
import torch.multiprocessing as mp
from torch.multiprocessing import Process
from multiprocessing.managers import SyncManager
from torch import set_num_threads, set_num_interop_threads
import psutil
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
import threading
from typing import Tuple, Callable, List, Dict, Any
from project.parallel_grid_search.code.parallel_utils import GenericJobGenerator, ComputeJobResourceManager, JobInterface
import errno
from os import waitpid, WNOHANG, environ
import torch

# prevent hpcx from affecting hpcy (shared memory)
import tempfile
node = environ.get("SLURMD_NODENAME") or environ.get("SLURM_NODELIST", "unknown")
tmp_dir = Path(f'/tmp/{node}')
tmp_dir.mkdir(exist_ok=True)
tempfile.tempdir = str(tmp_dir)
environ['TORCHINDUCTOR_CACHE_DIR'] = str(tmp_dir / 'torch_cache')

if mp.get_start_method(allow_none=True) is None:
    mp.set_start_method('spawn')

import logging
logger = logging.getLogger(__name__)


def worker_function(job_queue, result_queue, cores_per_job, priority, device):
    """Worker function - no shared state, just queues."""
    set_num_interop_threads(1)
    set_num_threads(cores_per_job)
    p = psutil.Process()
    
    if device.type == 'cuda':
        torch.cuda.set_device(device.index)
    
    try:
        p.nice(priority)
    except (psutil.AccessDenied, PermissionError):
        pass

    while True:
        try:
            job_data = job_queue.get(timeout=0.1)
            if job_data is None or job_data['job'] is None:
                break
        except queue.Empty:
            continue
        except KeyboardInterrupt:
            logger.debug(f"Worker interrupted with ctrl-c")
            raise
        except Exception:
            break
        
        job = job_data['job']
        start_time = time()
        result = job.run(device)
        result.update({
            'job': job,
            'device': device,
            'worker_pid': p.pid,
            'start_time': start_time,
            'stop_time': time(),
        })

        try:
            result_queue.put(result)
        except Exception:
            pass

        if hasattr(result['job'], 'cleanup'):
            result['job'].cleanup()

        logger.debug(f"Worker completed job {result['job'].i}-{result['job'].j} on {device}")


def worker_function_cpu(*args, **kwargs):
    # hide gpu stuff, but can still access torch.compile
    environ['CUDA_VISIBLE_DEVICES'] = ''
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
    """Worker pool with supervisor pattern - parent owns all state."""

    def __init__(self, max_workers, cores_per_job, process_priority=5, spawn_rate=1, device='cpu'):
        self.device = torch.device(device)
        self.max_workers = max_workers
        self.cores_per_job = cores_per_job
        self.process_priority = process_priority
        self.worker_job_queue = mp.Queue()
        self.result_queue = mp.Queue()
        self.spawn_delay = spawn_rate 
        self.accept_jobs = True
        
        # Parent-owned state — no SyncManager needed
        self.workers = dict()
        self._jobs_dispatched = 0
        self._jobs_returned = 0
        self._enabled = True
        
        self.name = device + '_pool'
        logger.info(f"{self.name} initialized with max_workers={max_workers}")
    
    @property
    def alive_workers(self):
        return sum(1 for p in self.workers.values() if p.is_alive())

    @property
    def total_workers(self):
        return self.alive_workers

    @property
    def busy_workers(self):
        return max(0, self._jobs_dispatched - self._jobs_returned)

    @property
    def backlog(self):
        return self.worker_job_queue.qsize()

    @property
    def idle_workers(self):
        return max(0, self.alive_workers - self.busy_workers)

    @property
    def jobs_completed(self):
        return self._jobs_returned

    @property
    def enabled(self):
        return self._enabled

    def enable(self):
        self._enabled = True

    def disable(self):
        self._enabled = False

    def scale_workers_to(self, target_workers: int):
        """Scale to target number of workers"""

        # forget dead workers - RIP
        for pid, process in list(self.workers.items()):
            if not process.is_alive():
                del self.workers[pid]

        current = self.alive_workers
        if target_workers > current:
            return self.scale_up_workers(target_workers - current)
        elif target_workers < current:
            return self.scale_down_workers(current - target_workers)
        return 0
            
    def scale_up_workers(self, count=1):
        """Add new workers up to max_workers limit"""
        added = 0
        current = self.alive_workers
        
        for _ in range(count):
            if current >= self.max_workers:
                break
            
            if self.idle_workers == 0:
                target = worker_function if self.device.type == 'cuda' else worker_function_cpu
                p = Process(
                    target=target,
                    args=(self.worker_job_queue, self.result_queue, self.cores_per_job, 
                        self.process_priority, self.device),
                    daemon=True,
                )
                p.start()
                self.workers[p.pid] = p
                current += 1
                added += 1
                
                logger.debug(f"{self.name}: Scaled up {added} workers...")
            else:
                logger.debug(f"{self.name}: Avoided scale up due to idle workers...")
            
            sleep(self.spawn_delay)
        
        return added

    def scale_down_workers(self, count=1):
        """Remove workers by sending shutdown signals to workers"""
        removed = 0
        for _ in range(min(count, self.alive_workers)):
            self.worker_job_queue.put(None, timeout=1)  # Shutdown signal
            removed += 1
        logger.debug(f"{self.name}: Scaled down {removed} workers (max: {self.max_workers}, job-device backlog: {self.backlog})")
        return -removed

    def submit_job(self, job):
        if self._enabled or job is None:
            self.worker_job_queue.put({'job': job})
            if job is not None:
                self._jobs_dispatched += 1
            return True
        else:
            return False
    
    def get_result(self, timeout=None):
        """Get a result from the result queue"""
        try:
            result = self.result_queue.get(timeout=timeout)
            self._jobs_returned += 1
            
            # Handle error: disable pool if worker reported error
            if result.get('status') == 'error' and result.get('error_type') != 'OOM':
                self._enabled = False
                
            return result
        except queue.Empty:
            return None
        except ValueError as e:
            if "is closed" in str(e):
                logger.debug(f"{self.name}: Result queue closed, exiting processing loop")
            else:
                logger.debug(f"{self.name}: Result queue Value Error")
            return None

    def shutdown(self):
        """Shutdown all workers and clean up resources"""
        logger.info(f"{self.name}: Initiating shutdown")
        self.softly_kill_all_workers()
        sleep(1)
        self.hard_kill_all_workers()
        
        self.worker_job_queue.close()
        self.result_queue.close()
        
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

    
class ResourceAwareScheduler:
    """Simplified scheduler with centralized resource management"""


    def __init__(self, resource_manager: ComputeJobResourceManager, manager: SyncManager, job_generator: GenericJobGenerator, scheduler_loop_delay:int=1):
        # Control
        self.resource_manager = resource_manager
        self.running = False
        self.submission_complete = False
        self.scheduler_thread = None
        self.scheduler_loop_delay = scheduler_loop_delay
        
        # Worker pool — no manager needed
        self.worker_pools = dict()
        pool = LazyWorkerPool(max_workers=self.resource_manager.max_concurrent, cores_per_job=self.resource_manager.cores_per_job, process_priority=15)
        self.worker_pools[pool.device] = pool
        if self.resource_manager.use_gpu:
            for gpu_id in self.resource_manager.gpu_manager.available_gpus:
                pool = LazyWorkerPool(max_workers=self.resource_manager.max_concurrent, cores_per_job=self.resource_manager.cores_per_job, process_priority=10, device=f'cuda:{gpu_id}')
                self.worker_pools[pool.device] = pool
        
        # Job tracking
        self.scheduler_job_queue = mp.Queue()
        self.job_completion_times = []  # Circular buffer for completion times
        self.max_completion_history = 20
        self.total_queued_jobs = 0
        self.last_completed_count = self.completed_count
        
        # Scaling parameters
        self.scaling_threshold = 1  # jobs per scaling
        self.job_buffer = int(round(self.target_concurrency * 1/4 + 0.5))

        self._start_job_submissions(job_generator)

    @property
    def target_concurrency_split_by_resource(self):
        return self.resource_manager.target_concurrency_split_by_resource 

    @property
    def target_concurrency(self):
        return sum(self.target_concurrency_split_by_resource.values()) 
    
    @property
    def jobs_in_queue(self):
        return self.scheduler_job_queue.qsize()

    @property
    def busy_workers(self):
        return sum(pool.busy_workers for pool in self.worker_pools.values()) 

    @property
    def backlog(self):
        return sum(pool.backlog for pool in self.worker_pools.values()) 

    @property
    def completed_count(self):
        return sum(pool.jobs_completed for pool in self.worker_pools.values())

    @property
    def max_workers(self):
        return sum(pool.max_workers for pool in self.worker_pools.values())

    @property
    def average_completion_time(self):
        return sum(self.job_completion_times) / len(self.job_completion_times) if self.job_completion_times else self.scheduler_loop_delay
        
    def _start_job_submissions(self, job_generator):
        """Start scheduler + job submission"""
        self.running = True
        self._start_job_submission(job_generator)

        
        logger.debug(f"Added initial jobs and workers {self.target_concurrency_split_by_resource}, total: {self.target_concurrency}")
        self._schedule_jobs(n_jobs=self.target_concurrency)
        self.scale_workers()
        
        self.scheduler_thread = threading.Thread(target=self._scheduler_loop)
        self.scheduler_thread.start()
        
        logger.info("Scheduler started")
        return self.running
    
    def scale_workers(self):
        target_concurrency = self.target_concurrency_split_by_resource
        delay = self._calculate_spawn_delay() # Thundering herd prevention
        delta = 0
        for device, pool in self.worker_pools.items():
            pool.spawn_delay = delay
            target = target_concurrency[device]
            delta += pool.scale_workers_to(target)
        return delta, sum(target_concurrency.values())

    def stop(self):
        """Stop the scheduler and clean up all resources"""
        if not self.running:
            return
        self.running = False
            
        logger.info("Initiating scheduler shutdown")
        
        # Wait for scheduler thread
        if self.scheduler_thread and self.scheduler_thread.is_alive():
            self.scheduler_thread.join(timeout=5)
        
        if hasattr(self, 'submit_thread') and self.submit_thread.is_alive():
            self.submit_thread.join(timeout=5)
        
        # Drain job queue
        drain_queue(self.scheduler_job_queue)
        
        # Shutdown worker pools
        for device, pool in self.worker_pools.items():
            pool.shutdown()
        
        logger.info("Scheduler shutdown complete")
        
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
       if exc_type is not None:
           logger.error(f"Exiting due to exception: {exc_type.__name__}: {exc_val}")
       self.stop()
       return False

    def _add_job(self, job, max_queue_size=100):
        while self.scheduler_job_queue.qsize() >= max_queue_size:
            if not self.running:
                return False
            sleep(5)
        self.scheduler_job_queue.put(job)
        self.total_queued_jobs += 1
        return True

    def _start_job_submission(self, job_generator):
        """Start background job submission thread"""
        def submit_jobs():
            for job in job_generator:
                if not self._add_job(job):
                    break
            self.submission_complete = True
        self.submit_thread = threading.Thread(target=submit_jobs, daemon=True)
        logger.info(f"Submitting {len(job_generator)} jobs to scheduler")
        self.submit_thread.start()

    def _calculate_spawn_delay(self):
        # Fast submission when under capacity
        utilization = self.busy_workers / max(1, self.target_concurrency)
                
        # Only throttle when near/over capacity
        base_delay = self.average_completion_time / max(1, self.target_concurrency)

        if utilization < 0.8:  # Under 80% capacity
            return base_delay * torch.rand(()).item() # Little delay - fill up quickly!
        # Scale delay based on how far over capacity we are
        if utilization > 1.0:
            return base_delay * (2 ** (utilization - 1))
        else:
            return base_delay * 0.5 

    def _handle_job_completion(self, result):
        """Handle job completion - release resources (if a job was launched resources were reserved)"""
        job = result['job']
        device = result['device']
        
        if result['status'] == 'completed':
            # Track completion time
            completion_time = result['stop_time'] - result['start_time']
            self.job_completion_times.append(completion_time)
            if len(self.job_completion_times) > self.max_completion_history:
                self.job_completion_times.pop(0)
            logger.info(f"Completed job {job.i}-{job.j} on device {device}")
            
        # Handle errors: known errors → retry job
        elif result['status'] == 'error':
            if result.get('error_type') == 'OOM':
                logger.error(f"OOM job {job.i}-{job.j} on device {device}")
                self._handle_oom(device, job)
            else:
                logger.error(f"Unknown error on job {job.i}-{job.j} on device {device}")

        self.resource_manager.release_resources(device)

    def _handle_oom(self, device, job: JobInterface):
        pool = self.worker_pools[device]
        pool.disable()
        self.resource_manager.handle_oom(pool.device)
        pool.enable()
        pool.submit_job(job)
        logger.info(f"{pool.name} - {pool.device} recovered from OOM")
    
    def _schedule_jobs(self, n_jobs=1):
        if not self.running:
            return
        for _ in range(n_jobs):
            try:
                job = self.scheduler_job_queue.get(timeout=0.1)
                if self._try_schedule_job(job):
                    logger.info(f"Scheduled job {job.i}-{job.j}")
                else:
                    logger.debug(f"Failed to schedule job {job.i}-{job.j}")
                    # Put back in queue
                    self.scheduler_job_queue.put(job)
                    break
            except queue.Empty:
                pass

    def _try_schedule_job(self, job):
        device = self.resource_manager.try_allocate_resources()
        if device is None:
            return False
        
        pool = self.worker_pools[device]
        if not pool.submit_job(job):
            self.resource_manager.release_resources(device)
            return False
        return True

    def _adjust_concurrency(self):
        """Adjust concurrency based on completion rate and system resources"""
        completed_count = self.completed_count
        jobs_completed_in_interval = completed_count - self.last_completed_count

        if jobs_completed_in_interval > 0:
            # adjust target concurrency
            self.resource_manager.adjust_target_concurrency()
            self.job_buffer = int(round(self.target_concurrency + 0.5)) + 1
            # propagate changes to worker pools
            self.scale_workers()
        
        if self.resource_manager.gpu_manager:
            logger.info(self.resource_manager.gpu_manager.get_status())
        logger.info(self.resource_manager.cpu_manager.get_status())
        self.last_completed_count = completed_count
        self.scaling_threshold = max(1, self.target_concurrency // 3)

    def _scheduler_loop(self):
        """Main scheduler loop"""
        logger.info(f"Jobs in queue: {self.jobs_in_queue}")
        logger.info(f"Busy workers: {self.busy_workers}")
        
        try:
            while self.running:
                # Schedule new jobs
                zombie_reaping()
                busy_workers, backlog, job_buffer = self.busy_workers, self.backlog, self.job_buffer
                n_jobs = self.target_concurrency + job_buffer - backlog - busy_workers
                logger.debug(f'Schedule more? target: {self.target_concurrency_split_by_resource}, buffer: {job_buffer}, jobs-in-worker-backlog: {backlog}, busy-workers: {busy_workers} -> ' + (f'adding {n_jobs}' if n_jobs > 0 else f'no action {n_jobs}'))
                if n_jobs > 0 and self.running:
                    self._schedule_jobs(n_jobs=n_jobs)

                for pool in self.worker_pools.values():
                    while True:
                        result = pool.get_result(timeout=0.1)
                        if result is None:
                            break
                        self._handle_job_completion(result)

                # Adjust concurrency based on completion rate
                if (self.completed_count - self.last_completed_count >= self.scaling_threshold) and self.running:
                    self._adjust_concurrency()

                if self.submission_complete and (self.completed_count >= self.total_queued_jobs):
                    logger.info("Scheduler loop exit condition met")
                    break  # Exit loop, cleanup will happen in finally
                else:
                    sleep(self.scheduler_loop_delay)

        except Exception as e:
            logger.exception(f"Scheduler loop error: {e}")
        finally:
            logger.info("Scheduler loop exiting - draining remaining results")
            # Drain all remaining results
            while self.busy_workers > 0:
                for pool in self.worker_pools.values():
                    result = pool.get_result(timeout=1.0)
                    if result is not None:
                        self._handle_job_completion(result)
                zombie_reaping()
                sleep(0.1)
    
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
    ) -> Tuple[List[Dict], Dict]:
    
    
    """
    Generic parallel grid search that works with any job type.
    
    Args:
        job_factory: Function that creates jobs (passed to GenericJobGenerator)
        total_configs: Number of configurations to test
        samples_per_config: Number of samples per configuration
        output_path: Path to save results
        save_config: Callback to save configuration
        process_results: Callback to process results
        gpu_memory_per_job_gb: GPU memory per job
        cpu_memory_per_job_gb: CPU memory per job
        cpu_cores_per_job: CPU cores per job
        
    Returns:
        Path
    """
    
    # Create output directory
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=False)
    
    # Save configuration
    save_config(output_path)
    
    # Initialize scheduler
    resource_manager = ComputeJobResourceManager(
        cpu_memory_per_job_gb=cpu_memory_per_job_gb,
        cpu_cores_per_job=cpu_cores_per_job,
        gpu_memory_per_job_gb=gpu_memory_per_job_gb,
    )

    manager = mp.Manager()

    def write_history(locks, shared, flush=False, tresh=history_write_thresh):
        with locks['history']:
            if len(shared['history']) < tresh and not flush:
                return
            if len(shared['history']) == 0:
                return
            entries = list(shared['history'])
            shared['history'][:] = []
        process_results(entries, output_path, flush)
        logger.info(f"Wrote {len(entries)} results to {output_path}")
    try:
        job_generator = GenericJobGenerator(manager=manager,
                                 job_factory=job_factory,
                                 total_configs=total_configs,
                                 samples_per_config=samples_per_config,
                                 )
        shared = job_generator.shared 
        locks = job_generator.locks
        with ResourceAwareScheduler(resource_manager=resource_manager, manager=manager, job_generator=job_generator) as scheduler:
            start_time = time()
            with logging_redirect_tqdm():
                total_jobs = len(job_generator)
                with tqdm(total=total_jobs, desc="Grid Search Progress", unit="jobs") as pbar:
                    while pbar.n < total_jobs:
                        pbar.n = scheduler.completed_count
                        pbar.refresh()
                        write_history(locks, shared)
                        sleep(0.5)
            write_history(locks, shared, flush=True)
            if cleanup is not None:
                cleanup()
                
        # Process results with provided callback
        elapsed_time = time() - start_time
        logger.info(f"Grid search completed in {elapsed_time:.1f}s")
        
    except Exception as e:
        logger.exception(f"Unexpected error in grid search: {e}")
        raise
    finally:
        logger.info("Shutting down shared manager")
        manager.shutdown()