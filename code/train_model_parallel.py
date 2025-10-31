from pathlib import Path
from time import sleep, time
import queue
import torch.multiprocessing as mp
from torch.multiprocessing import Process, Queue
from torch import set_num_threads, set_num_interop_threads
import psutil
import pandas as pd
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
import threading
from typing import Tuple, Callable, List, Dict, Any
from projects.parallel_grid_search.code.parallel_utils import GenericJobGenerator, ComputeJobResourceManager
import errno
from os import waitpid, WNOHANG
from copy import deepcopy

import logging
logger = logging.getLogger(__name__)

'''
TODO
reduce complexity / redundance + make more modular
optimizer for scheduler to find settings that maximize throughput: type cpu/gpu, concurrency, number of cores, throughput
'''

def worker_function(job_queue, result_queue, cores_per_job, priority, worker_stats, stats_lock):
    """Worker with proper stat tracking"""
    set_num_interop_threads(1)
    set_num_threads(cores_per_job)
    p = psutil.Process()
    # Set process priority to prevent freezing and prioritized gpu jobs over cpu jobs
    try:
        p.nice(priority)
    except (psutil.AccessDenied, PermissionError):
        pass

    worker_registered = False
    try:
        # Register worker as idle
        with stats_lock:
            worker_stats['idle'] += 1
            worker_registered = True
        
        while True:
            # Worker is IDLE while waiting for jobs
            job_data = None
            try:
                job_data = job_queue.get(timeout=10)
            except queue.Empty:
                break
                    
            if job_data is None:  # Shutdown signal
                break
                
            # NOW we transition from idle -> busy (only when we have a job to execute)
            with stats_lock:
                worker_stats['idle'] -= 1
                worker_stats['busy'] += 1
            
            job_completed_successfully = False
            try:
                job = job_data['job']
                resources = job_data['resources']
                device = resources['device']

                # Run job (this is when we're actually "busy")
                start_time = time()
                result = job.run(device)
                zombie_reaping() # i.ex dataloader may spawn many threads
                
                # Add metadata
                result.update({
                    'status': 'completed', 
                    'job': job,
                    'device': device,
                    'worker_pid': p.pid,
                    'start_time': start_time,
                    'stop_time': time(),
                })
                
                job_completed_successfully = True
                result_queue.put(result)
                logger.debug(f"Worker completed job {job.i}-{job.j} on {device}")
                
            except Exception as e:
                logger.exception(f"Job execution error: {e}")
                # Send error result back
                error_result = {
                    'status': 'error', 
                    'error_type': 'general', 
                    'error': str(e),
                    'job': job if 'job' in locals() else None,
                    'device': device if 'device' in locals() else None,
                    'worker_pid': p.pid
                }
                result_queue.put(error_result)
                
            finally:
                # Always transition back to idle
                with stats_lock:
                    worker_stats['busy'] -= 1
                    worker_stats['idle'] += 1
                    # Only increment completed if job actually succeeded
                    if job_completed_successfully:
                        worker_stats['jobs_completed'] += 1
                    else:
                        worker_stats['jobs_failed'] += 1
    except KeyboardInterrupt:
        logger.info("Worker received KeyboardInterrupt, shutting down")
    except Exception as e:
        logger.exception(f"Worker error: {e}")
    finally:
        # Clean up: remove this worker from stats
        if worker_registered:
            with stats_lock:
                worker_stats['idle'] -= 1  # Worker was idle when it shut down
        logger.info("Worker shut down")

def zombie_reaping():
    while True:
        try:
            pid, _ = waitpid(-1, WNOHANG)
            if pid == 0:
                break
        except OSError as e:
            if e.errno == errno.ECHILD:
                break

class LazyWorkerPool:
    """Simplified worker pool that creates workers on demand"""

    def __init__(self, max_workers: int, cores_per_job: int, process_priority=5, name=None, spawn_rate=1):
        # Don't force set_start_method here - it should be set once globally
        # mp.set_start_method('spawn', force=True)  # REMOVE THIS
        
        self.max_workers = max_workers
        self.cores_per_job = cores_per_job
        self.process_priority = process_priority
        self.worker_job_queue = Queue()
        self.result_queue = Queue()
        self.spawn_delay = spawn_rate 
        
        # Each pool gets its OWN manager
        self.mp_manager = mp.Manager()
        self.worker_stats = self.mp_manager.dict({
            'idle': 0,
            'busy': 0,
            'jobs_completed': 0,
            'jobs_failed': 0,
        })
        self.stats_lock = self.mp_manager.Lock()
        self.workers = dict()
        
        self.name = self.__class__.__name__ if name is None else name
        logger.info(f"{self.name} initialized with max_workers={max_workers}")
    
    def safe_get_stat(self, key):
        with self.stats_lock:
            return self.worker_stats[key]

    @property
    def total_workers(self):
        return self.busy_workers + self.idle_workers

    @property
    def busy_workers(self):
        return self.safe_get_stat('busy')

    @property
    def backlog(self):
        return self.worker_job_queue.qsize()

    @property
    def idle_workers(self):
        return self.safe_get_stat('idle')

    @property
    def jobs_completed(self):
        return self.safe_get_stat('jobs_completed')

    def scale_workers_to(self, target_workers: int):
        """Scale to target number of workers"""

        # forget dead workers - RIP
        for pid, process in list(self.workers.items()):
            if not process.is_alive():
                del self.workers[pid]

        current = self.total_workers
        if target_workers > current:
            return self.scale_up_workers(target_workers - current)
        elif target_workers < current:
            return self.scale_down_workers(current - target_workers)
        return 0
            
    def scale_up_workers(self, count=1):
        """Add new workers up to max_workers limit"""
        added = 0
        current = self.total_workers
        idle_count = self.idle_workers 

        for _ in range(count):
            if current >= self.max_workers:
                break
            
            if idle_count == 0:
                # Create new worker with shared state
                p = Process(
                    target=worker_function,
                    args=(self.worker_job_queue, self.result_queue, self.cores_per_job, self.process_priority, self.worker_stats, self.stats_lock)
                )
                p.start()
                self.workers[p.pid] = p
                current += 1
                added += 1
                
                logger.info(f"{self.name}: Scaled up {added} workers - {self.worker_stats} (max: {self.max_workers}, job-device backlog: {self.backlog})")
            else:
                logger.info(f"{self.name}: Avoided scale up due to idle workers - {self.worker_stats} (max: {self.max_workers}, job-device backlog: {self.backlog})")
            sleep(self.spawn_delay)
        return added

    def scale_down_workers(self, count=1):
        """Remove workers by sending shutdown signals to workers"""
        removed = 0
        for _ in range(min(count, self.total_workers)):
            self.worker_job_queue.put(None, timeout=1)  # Shutdown signal
            removed += 1
        logger.info(f"{self.name}: Scaled down {removed} workers - {self.worker_stats} (max: {self.max_workers}, job-device backlog: {self.backlog})")
        return -removed
    
    def submit_job(self, job, device):
        """Submit a job with pre-allocated device"""
        job_data = {
            'job': job,
            'resources': {
                'device': device, 
            },
        }
        self.worker_job_queue.put(job_data)
   
    def get_result(self, timeout=None):
        """Get a result from the result queue"""
        try:
            result = self.result_queue.get(timeout=timeout)
            return result
        except queue.Empty:
            return None
        except ValueError as e:
            if "is closed" in str(e):
                logger.debug(f"{self.name}: Result queue closed, exiting processing loop")
            else:
                logger.debug(f"{self.name}: Result queue Value Error")
            return None

    def shutdown(self, timeout=30):
        """Shutdown all workers and clean up resources"""
        logger.info(f"{self.name}: Initiating LazyWorkerPool shutdown")
        logger.info(f"{self.name}: Workers alive: {len(self.workers)}, Stats: {dict(self.worker_stats)}")
        logger.info(f"{self.name}: Queue sizes - worker_job: {self.worker_job_queue.qsize()}, result: {self.result_queue.qsize()}")
        
        # 1. Send shutdown signals
        num_signals = len(self.workers)
        logger.info(f"{self.name}: Sending {num_signals} shutdown signals")
        for i in range(num_signals):
            try:
                self.worker_job_queue.put(None, timeout=1)
                logger.info(f"{self.name}: Sent shutdown signal {i+1}/{num_signals}")
            except:
                logger.info(f"{self.name}: Failed to send shutdown signal {i+1}")
        
        # 2. Wait for workers to exit gracefully
        logger.info(f"{self.name}: Waiting for workers to exit gracefully (timeout={timeout}s)")
        start = time()
        for pid, process in list(self.workers.items()):
            remaining = timeout - (time() - start)
            if remaining <= 0:
                logger.info(f"{self.name}: Timeout reached, {len(self.workers)} workers still alive")
                break
            logger.info(f"{self.name}: Waiting for worker {pid} (remaining={remaining:.1f}s)")
            process.join(timeout=min(remaining, 2))
            if not process.is_alive():
                logger.info(f"{self.name}: Worker {pid} exited cleanly")
                del self.workers[pid]
            else:
                logger.info(f"{self.name}: Worker {pid} still alive after join")
        
        logger.info(f"{self.name}: After graceful wait: {len(self.workers)} workers still alive")
        
        # 3. Force terminate stragglers
        if self.workers:
            logger.info(f"{self.name}: Force terminating {len(self.workers)} remaining workers")
            for pid, process in list(self.workers.items()):
                if process.is_alive():
                    logger.info(f"{self.name}: Terminating worker {pid}")
                    process.terminate()
                    process.join(timeout=2)
                    if process.is_alive():
                        logger.info(f"{self.name}: Killing worker {pid}")
                        process.kill()
                        process.join(timeout=1)
            self.workers.clear()
            logger.info(f"{self.name}: All workers terminated")
        
        # 4. Drain queues
        logger.info(f"{self.name}: Draining result queue (size={self.result_queue.qsize()})")
        drained = 0
        while not self.result_queue.empty():
            try:
                self.result_queue.get_nowait()
                drained += 1
            except:
                break
        logger.info(f"{self.name}: Drained {drained} results from queue")
        
        # 5. Close queues
        logger.info(f"{self.name}: Closing queues")
        self.worker_job_queue.cancel_join_thread()
        self.result_queue.cancel_join_thread()
        self.worker_job_queue.close()
        self.result_queue.close()
        logger.info(f"{self.name}: Queues closed")
        
        # 6. Shutdown manager
        logger.info(f"{self.name}: Final stats before manager shutdown: {dict(self.worker_stats)}")
        logger.info(f"{self.name}: Shutting down multiprocessing manager")
        self.mp_manager.shutdown()
        logger.info(f"{self.name}: Manager shutdown complete")
        try:
            self.mp_manager.shutdown()
            logger.info(f"{self.name}: Manager shutdown complete")
        except Exception as e:
            logger.error(f"{self.name}: Manager shutdown failed: {e}")
            raise

        logger.info(f"{self.name}: shutdown complete")

    def shutdown2(self, timeout=30):
        """Shutdown all workers and clean up resources"""
        logger.info(f"{self.name}: Initiating LazyWorkerPool shutdown")
        
        # 1. Send shutdown signals (one per worker)
        num_signals = len(self.workers)
        for _ in range(num_signals):
            try:
                self.worker_job_queue.put(None, timeout=1)
            except:
                pass
        
        # 2. Wait for workers to exit gracefully
        start = time()
        for pid, process in list(self.workers.items()):
            remaining = timeout - (time() - start)
            if remaining <= 0:
                break
            process.join(timeout=min(remaining, 2))
            if not process.is_alive():
                del self.workers[pid]
        
        # 3. Force terminate stragglers
        for pid, process in list(self.workers.items()):
            if process.is_alive():
                logger.warning(f"{self.name}: Force terminating worker {pid}")
                process.terminate()
                process.join(timeout=2)
                if process.is_alive():
                    process.kill()
        self.workers.clear()
        
        # 4. Drain queues BEFORE closing
        while not self.result_queue.empty():
            try:
                self.result_queue.get_nowait()
            except:
                break
        
        # 5. Close queues properly
        self.worker_job_queue.cancel_join_thread()
        self.result_queue.cancel_join_thread()
        self.worker_job_queue.close()
        self.result_queue.close()
        
        # 6. Shutdown manager
        self.mp_manager.shutdown()
        
        logger.info(f"{self.name}: shutdown complete")    

class ResourceAwareScheduler:
    """Simplified scheduler with centralized resource management"""
    
    def __init__(self, resource_manager: ComputeJobResourceManager, scheduler_loop_delay:int=1):
        # Control
        self.resource_manager = resource_manager
        self.running = False
        self.scheduler_thread = None
        self.scheduler_loop_delay = scheduler_loop_delay
        
        # Worker pool
        self.cpu_worker_pool = LazyWorkerPool(max_workers=self.resource_manager.max_concurrent, cores_per_job=self.resource_manager.cores_per_job, process_priority=5, name='cpu')
        self.gpu_worker_pool = LazyWorkerPool(max_workers=self.resource_manager.max_concurrent, cores_per_job=self.resource_manager.cores_per_job, process_priority=10, name='gpu')
        self.worker_pools = {'cpu': self.cpu_worker_pool, 'gpu': self.gpu_worker_pool}
        
        # Job tracking
        self.job_queue = Queue()
        self.job_completion_times = []  # Circular buffer for completion times
        self.max_completion_history = 20
        self.total_queued_jobs = 0
        self.last_completed_count = self.completed_count
        
        # Scaling parameters
        self.scaling_threshold = 1  # jobs per scaling
        self.job_buffer = int(round(self.target_concurrency * 1/4 + 0.5))

    @property
    def target_concurrency_split_by_resource(self):
        return self.resource_manager.target_concurrency_split_by_resource 

    @property
    def target_concurrency(self):
        return sum(self.target_concurrency_split_by_resource.values()) 
    
    @property
    def jobs_in_queue(self):
        return self.job_queue.qsize()

    @property
    def jobs_in_flight(self):
        return self.cpu_worker_pool.busy_workers + self.gpu_worker_pool.busy_workers

    @property
    def backlog(self):
        return self.cpu_worker_pool.backlog + self.gpu_worker_pool.backlog

    @property
    def completed_count(self):
        return self.cpu_worker_pool.jobs_completed + self.gpu_worker_pool.jobs_completed

    @property
    def max_workers(self):
        return self.cpu_worker_pool.max_workers + self.gpu_worker_pool.max_workers

    @property
    def average_completion_time(self):
        return sum(self.job_completion_times) / len(self.job_completion_times) if self.job_completion_times else self.scheduler_loop_delay
        
    def start(self):
        """Start scheduler"""
        self.running = True
        
        logger.debug(f"Added initial jobs and workers {self.target_concurrency_split_by_resource}, total: {self.target_concurrency}")
        self._schedule_jobs(n_jobs=self.target_concurrency)
        self.scale_workers()
        
        self.scheduler_thread = threading.Thread(target=self._scheduler_loop)
        self.scheduler_thread.start()
        
        logger.info("Scheduler started")
        return self.running
    
    def scale_workers(self):
        target_concurrency = self.target_concurrency_split_by_resource
        target_gpu = target_concurrency['gpu']
        target_cpu = target_concurrency['cpu']
        delay = self._calculate_spawn_delay() # Thundering herd prevention
        self.gpu_worker_pool.spawn_delay = delay
        self.cpu_worker_pool.spawn_delay = delay
        gpu_delta = self.gpu_worker_pool.scale_workers_to(target_gpu)
        cpu_delta = self.cpu_worker_pool.scale_workers_to(target_cpu)
        return gpu_delta + cpu_delta, target_gpu + target_cpu 

    def stop(self):
        """Stop the scheduler and clean up all resources"""
        if not self.running:
            return
            
        logger.info("Initiating scheduler shutdown")
        self.running = False
        
        # Wait for scheduler loop to exit FIRST
        if self.scheduler_thread and self.scheduler_thread.is_alive():
            logger.info("Waiting for scheduler thread to terminate")
            self.scheduler_thread.join(timeout=10)
            if self.scheduler_thread.is_alive():
                logger.warning("Scheduler thread did not terminate within timeout")
        
        logger.info("Scheduler thread joined, starting cleanup")
        zombie_reaping()
        logger.info("Zombie reaping complete")

        # Close job queue
        try:
            logger.info("Closing job queue")
            self.job_queue.close()
            self.job_queue.cancel_join_thread()
            logger.info("Job queue closed")
        except Exception as e:
            logger.error(f"Error closing job queue: {e}")

        # Delegate worker pool shutdown
        try:
            logger.info("Shutting down CPU worker pool")
            self.cpu_worker_pool.shutdown()
            logger.info("CPU worker pool shutdown complete")
            
            logger.info(f"GPU worker pool state: workers={len(self.gpu_worker_pool.workers)}, stats={dict(self.gpu_worker_pool.worker_stats)}")
            logger.info("Shutting down GPU worker pool")
            self.gpu_worker_pool.shutdown()
            logger.info("GPU worker pool shutdown complete")
        except Exception as e:
            logger.error(f"Error shutting down worker pool: {e}")

        logger.info("Scheduler shutdown complete")

    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        return False

    def add_job(self, job):
        """Add job to queue"""
        self.job_queue.put(job)
        self.total_queued_jobs += 1

    def _calculate_spawn_delay(self):
        # Fast submission when under capacity
        utilization = self.jobs_in_flight / max(1, self.target_concurrency)
        
        if utilization < 0.8:  # Under 80% capacity
            return 0  # No delay - fill up quickly!
        
        # Only throttle when near/over capacity
        base_delay = self.average_completion_time / max(1, self.target_concurrency)
        
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
            
        # Handle errors
        elif result['status'] == 'error':
            if result.get('error_type') == 'OOM':
                logger.error(f"OOM job {job.i}-{job.j} on device {device}")
                self.resource_manager.handle_oom(device)
            else:
                logger.error(f"Unknown error on job {job.i}-{job.j} on device {device}")
            logger.warning(f"Putting {job.i}-{job.j} back in job_queue")

        self.resource_manager.release_resources(device)

    def _schedule_jobs(self, n_jobs=1):
        for _ in range(n_jobs):
            try:
                job = self.job_queue.get(timeout=0.1)
                if self._try_schedule_job(job):
                    logger.info(f"Scheduled job {job.i}-{job.j}")
                else:
                    logger.debug(f"Failed to schedule job {job.i}-{job.j}")
                    # Put back in queue
                    self.job_queue.put(job)
                    break
            except queue.Empty:
                pass

    def _try_schedule_job(self, job):
        """Try to schedule a single job, returns True if scheduled, False otherwise"""
        device = self.resource_manager.try_allocate_resources()

        if device is not None:
            is_gpu = device.type == 'cuda'
            try:
                logger.debug(f"Scheduled job on device {device}")
                if is_gpu:
                    self.gpu_worker_pool.submit_job(job, device)
                else:
                    self.cpu_worker_pool.submit_job(job, device)
                
                return True
                
            except Exception as e:
                # Job submission failed - must release resources
                logger.error(f"Failed to submit job, releasing resources: {e}")
                self.resource_manager.release_resources(device)
                return False
        return False

    def _adjust_concurrency(self):
        """Adjust concurrency based on completion rate and system resources"""
        completed_count = self.completed_count
        jobs_completed_in_interval = completed_count - self.last_completed_count

        if completed_count % 10 == 0 and completed_count > 0:
            logger.info(f"Progress: {completed_count}/{self.total_queued_jobs} jobs completed")

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
        self.scaling_threshold = self.target_concurrency 

    def _scheduler_loop(self):
        """Main scheduler loop"""
        logger.info(f"Jobs in queue: {self.jobs_in_queue}")
        logger.info(f"Jobs in flight: {self.jobs_in_flight}")
        
        try:
            while self.running:
                # Schedule new jobs
                zombie_reaping() # i.ex dataloader may spawn many threads
                in_flight, backlog, job_buffer = self.jobs_in_flight, self.backlog, self.job_buffer
                n_jobs = self.target_concurrency + job_buffer - backlog - in_flight
                logger.debug(f'Schedule more? target: {self.target_concurrency_split_by_resource}, buffer: {job_buffer}, jobs-in-worker-backlog: {backlog}, in-flight: {in_flight} -> ' + (f'adding {n_jobs}' if n_jobs > 0 else f'no action {n_jobs}'))
                if n_jobs > 0 and self.running:
                    self._schedule_jobs(n_jobs=n_jobs)

                for worker_pool in self.worker_pools.values():
                    while self.running:
                        result = worker_pool.get_result(timeout=0.1)
                        if result is None or not self.running:
                            break
                        self._handle_job_completion(result)

                # Adjust concurrency based on completion rate
                if (self.completed_count - self.last_completed_count >= self.scaling_threshold) and self.running:
                    self._adjust_concurrency()

                if self.completed_count >= self.total_queued_jobs:
                    self.running = False
                else:
                    sleep(self.scheduler_loop_delay)

        except Exception as e:
            logger.exception(f"Scheduler loop error: {e}")
        finally:
            logger.info("Scheduler loop exiting")
            self.running = False

def generic_parallel_grid_search(
    # Core parameters
    job_factory: Callable,
    total_configs: int,
    samples_per_config: int,
    output_path: Path,
    save_config: Callable[[Path], None],
    process_results: Callable[[List[Dict], Dict, Path], Any],
    # Resource parameters
    gpu_memory_per_job_gb: float = None,
    cpu_memory_per_job_gb: float = None,
    cpu_cores_per_job: int = 1,
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
        Tuple of (history, best_params)
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

    try:
        with GenericJobGenerator(job_factory=job_factory,
                                 total_configs=total_configs,
                                 samples_per_config=samples_per_config,
        ) as job_generator:
            history = None
            best_params = None
            with ResourceAwareScheduler(resource_manager=resource_manager) as scheduler:
                total_jobs = len(job_generator)
                logger.info(f"Submitting {total_jobs} jobs to scheduler")
                
                for job in job_generator:
                    scheduler.add_job(job)
                
                scheduler.start()
                
                # Wait for completion
                start_time = time()
                with logging_redirect_tqdm():
                    with tqdm(total=total_jobs, desc="Grid Search Progress", unit="job") as pbar:
                        last_count = 0
                        while scheduler.completed_count < total_jobs and scheduler.running:
                            current_count = scheduler.completed_count
                            if current_count > last_count:
                                pbar.update(current_count - last_count)
                                last_count = current_count
                            sleep(0.5)
                
                if scheduler.completed_count == total_jobs:
                    elapsed_time = time() - start_time
                    logger.info(f"Grid search completed in {elapsed_time:.1f}s")
                logger.info("Exiting scheduler context manager")
                sleep(3)
                    
            # Extract results from shared state
            logger.info("Scheduler shutdown complete")
            sleep(3)

            shared = job_generator.shared 
            locks = job_generator.locks
            with locks.get('history', locks):
                history = list(shared.get('history', []))
                # Process results with provided callback
                logger.info(f"Active threads before sleep: {threading.active_count()}")
                sleep(3)
                logger.info(f"Active threads before processing: {threading.active_count()}")
                process_results(history, best_params, output_path)
                logger.info(f"Saved {len(history)} results to {output_path}")

            logger.info("Exiting job generator context manager")

        logger.info("Job generator context exited")

        
        return history, best_params
                    
    except KeyboardInterrupt:
        logger.info("Main function received KeyboardInterrupt, ensuring cleanup")
        raise
    except Exception as e:
        logger.exception(f"Unexpected error in grid search: {e}")
        raise