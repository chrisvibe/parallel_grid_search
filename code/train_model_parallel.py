import random
from pathlib import Path
import time
import queue
import torch.multiprocessing as mp
from torch.multiprocessing import Process, Queue
from torch import set_num_threads, set_num_interop_threads
import psutil
import pandas as pd
from tqdm import tqdm
import threading
from typing import Tuple, Callable, List, Dict, Any
from projects.parallel_grid_search.code.parallel_utils import GenericJobGenerator, ComputeJobResourceManager
import sys

import logging
logger = logging.getLogger(__name__)

'''
TODO
set dataset_cache limit dynamically
not sure dataset_cache memory scheme is optimal
reduce complexity / redundance + make more modular
optimizer for scheduler to find settings that maximize throughput: type cpu/gpu, concurrency, number of cores, throughput
split into cpu and gpu queue since they may require different resources
'''

def worker_function(job_queue, result_queue, worker_stats, stats_lock):
    """Worker with proper stat tracking"""
    p = psutil.Process()
    worker_registered = False
    resources_limited = False
    try:
        # Register worker as idle
        with stats_lock:
            worker_stats['total'] += 1
            worker_stats['idle'] += 1
            worker_registered = True
        
        while True:
            # Worker is IDLE while waiting for jobs
            job_data = None
            try:
                job_data = job_queue.get(timeout=1)
            except queue.Empty:
                continue  # Still idle, waiting for jobs
                
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
                resources_allocated = resources.get('resources_allocated', False)
                device = resources['device']
                cores_per_job = resources['cores_per_job']

                if not resources_limited:
                    # limit resources per process
                    set_num_interop_threads(1)
                    set_num_threads(cores_per_job)
                    resources_limited = True
                    
                # Set process priority to prevent freezing and prioritized gpu jobs over cpu jobs
                is_gpu = device.type == 'cuda'
                try:
                    p.nice(5 if is_gpu else 10)
                except (psutil.AccessDenied, PermissionError):
                    pass
                
                # Run job (this is when we're actually "busy")
                start_time = time.time()
                result = job.run(device)
                
                # Add metadata
                result.update({
                    'resources_allocated': resources_allocated,
                    'job': job,
                    'device': device,
                    'worker_pid': p.pid,
                    'start_time': start_time,
                    'stop_time': time.time(),
                })
                
                job_completed_successfully = True
                result_queue.put(result)
                logger.debug(f"Worker completed job {job.i}-{job.j} on {device}")
                
            except Exception as e:
                logger.exception(f"Job execution error: {e}")
                # Send error result back
                error_result = {
                    'resources_allocated': resources.get('resources_allocated', False) if 'resources' in locals() else False,
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
                worker_stats['total'] -= 1
                worker_stats['idle'] -= 1  # Worker was idle when it shut down
        logger.info("Worker shut down")


class LazyWorkerPool:
    """Simplified worker pool that creates workers on demand"""

    def __init__(self, max_workers: int):
        mp.set_start_method('spawn', force=True)
        self.max_workers = max_workers
        self.worker_job_queue = Queue()
        self.result_queue = Queue()

        self.mp_manager = mp.Manager()
        self.worker_stats = self.mp_manager.dict({
            'idle': 0,
            'busy': 0,
            'total': 0,
            'jobs_completed': 0,
            'jobs_failed': 0,
        })
        self.stats_lock = self.mp_manager.Lock()
        self.workers = dict() # pid -> process
        
        logger.info(f"LazyWorkerPool initialized with max_workers={max_workers}")
    
    def safe_get_stat(self, key):
        with self.stats_lock:
            return self.worker_stats[key]

    @property
    def get_total_workers(self):
        return self.safe_get_stat('total')

    @property
    def get_busy_workers(self):
        return self.safe_get_stat('busy')

    @property
    def get_jobs_in_flight(self):
        return self.get_busy_workers

    @property
    def get_idle_workers(self):
        return self.safe_get_stat('idle')

    @property
    def get_jobs_completed(self):
        return self.safe_get_stat('jobs_completed')

    @property
    def backlog(self):
        return self.worker_job_queue.qsize()

    def scale_workers_to(self, target_workers: int):
        """Scale to target number of workers"""

        # forget dead workers - RIP
        for pid, process in list(self.workers.items()):
            if not process.is_alive():
                del self.workers[pid]

        current = self.get_total_workers
        if target_workers > current:
            return self.scale_up_workers(target_workers - current)
        elif target_workers < current:
            return self.scale_down_workers(current - target_workers)
            
    def scale_up_workers(self, count=1):
        """Add new workers up to max_workers limit"""
        added = 0
        current = self.get_total_workers
        idle_count = self.get_idle_workers 

        for _ in range(count):
            if current >= self.max_workers:
                break
            
            if idle_count == 0:
                # Create new worker with shared state
                p = Process(
                    target=worker_function,
                    args=(self.worker_job_queue, self.result_queue, self.worker_stats, self.stats_lock)
                )
                p.start()
                self.workers[p.pid] = p
                current += 1
                added += 1
                
                logger.info(f"Scaled up {added} workers (busy: {self.get_busy_workers}, idle: {self.get_idle_workers}, total: {self.get_total_workers}, max: {self.max_workers}, job-device backlog: {self.backlog})")
            else:
                logger.info(f"Avoided scale up due to idle workers (busy: {self.get_busy_workers}, idle: {self.get_idle_workers}, total: {self.get_total_workers}, max: {self.max_workers}, job-device backlog: {self.backlog})")
        return added

    def scale_down_workers(self, count=1):
        """Remove workers by sending shutdown signals to workers"""
        removed = 0
        for _ in range(min(count, self.get_total_workers)):
            self.worker_job_queue.put(None)  # Shutdown signal
            removed += 1
        logger.info(f"Scaled down {removed} workers (busy: {self.get_busy_workers}, idle: {self.get_idle_workers}, total: {self.get_total_workers}, max: {self.max_workers}, job-device backlog: {self.backlog})")
        return removed

    def submit_job(self, job, device, cores_per_job):
        """Submit a job with pre-allocated device"""
        job_data = {
            'job': job,
            'resources': {
                'resources_allocated': True,
                'device': device, 
                'cores_per_job': cores_per_job,
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
                logger.debug("Result queue closed, exiting processing loop")
            else:
                logger.debug("Result queue Value Error")
            return None
    
    def shutdown(self, timeout=30):
        """Shutdown all workers and clean up resources"""
        logger.info("Initiating LazyWorkerPool shutdown")
        logger.debug(f"Worker Job queue size before: {self.worker_job_queue.qsize()}")
        logger.debug(f"Result queue size before: {self.result_queue.qsize()}")
        logger.debug(f"Stats before: {self.worker_stats}")

        start_time = time.time()
        while self.get_total_workers > 0 and (time.time() - start_time) < timeout:
            # Send shutdown signals to all workers first
            for _ in range(self.get_total_workers):
                try:
                    self.worker_job_queue.put(None)
                except Exception as e:
                    logger.error(f"Error sending shutdown signal: {e}")
                logger.debug(f"Worker job queue size after signals: {self.worker_job_queue.qsize()}")

            # Wait for workers to process signals
            for pid, process in list(self.workers.items()):
                if process.is_alive():
                    logger.debug(f"Worker {pid} still alive, waiting...")
                    process.join(timeout=0.1)
                else:
                    del self.workers[pid]
            time.sleep(0.1)
        
        items = []
        while not self.worker_job_queue.empty():
            try:
                items.append(self.worker_job_queue.get_nowait())
            except:
                break
        logger.info(f"LazyWorkerPool queue {items}")

        # Force terminate any remaining workers
        for pid, process in list(self.workers.items()):
            if process.is_alive():
                logger.warning(f"Force terminating worker")
                process.terminate()
                try:
                    process.join(timeout=2)
                except:
                    process.kill()
        self.workers.clear()

        # Close queues
        try:
            self.worker_job_queue.close()
            self.result_queue.close()
        except Exception as e:
            logger.error(f"Error closing queues: {e}")

        # Shutdown the manager
        logger.debug(f"Stats after: {self.worker_stats}")
        try:
            self.mp_manager.shutdown()
            # Wait for the manager process to terminate gracefully
            self.mp_manager.join(timeout=2)
        except Exception as e:
            logger.error(f"Error shutting down manager: {e}")
            # Force kill if still alive
            try:
                if hasattr(self.mp_manager, '_process') and self.mp_manager._process.is_alive():
                    self.mp_manager._process.terminate()
                    self.mp_manager._process.join(timeout=1)
                    if self.mp_manager._process.is_alive():
                        self.mp_manager._process.kill()
            except Exception as e:
                logger.error(f"Error force-killing manager: {e}")

        logger.debug(f"Worker job queue size after: {self.worker_job_queue.qsize()}")
        logger.debug(f"Result queue size after: {self.result_queue.qsize()}")
        logger.info("LazyWorkerPool shutdown complete")
    

class ResourceAwareScheduler:
    """Simplified scheduler with centralized resource management"""
    
    def __init__(self, resource_manager: ComputeJobResourceManager, initial_concurrency=1, scheduler_loop_delay =1):
        # Control
        self.resource_manager = resource_manager
        self.running = False
        self.scheduler_thread = None
        self.scheduler_loop_delay = scheduler_loop_delay
        
        # Worker pool
        self.worker_pool = LazyWorkerPool(max_workers=self.resource_manager.max_concurrent)
        
        # Job tracking
        self.job_queue = Queue()
        self.job_completion_times = [30]  # Circular buffer for completion times (init with 30s default)
        self.max_completion_history = 20
        self.total_queued_jobs = 0
        self.last_completed_count = self.completed_count
        
        # Scaling parameters
        self.last_scaling_time = time.time()
        self.scaling_interval = 30  # seconds
        self.target_concurrency = initial_concurrency
        self.job_buffer = int(round(self.target_concurrency / 2 + 0.5))
    
    @property
    def jobs_in_queue(self):
        return self.job_queue.qsize()

    @property
    def jobs_in_flight(self):
        return self.worker_pool.get_jobs_in_flight

    @property
    def jobs_in_worker_backlog(self):
        return self.worker_pool.backlog

    @property
    def completed_count(self):
        return self.worker_pool.get_jobs_completed
        
    def start(self):
        """Start scheduler"""
        self.running = True
        
        logger.debug(f"Added initial jobs and workers {self.target_concurrency}")
        self._schedule_jobs(n_jobs=self.target_concurrency)
        self.worker_pool.scale_up_workers(self.target_concurrency)
        
        self.scheduler_thread = threading.Thread(target=self._scheduler_loop)
        self.scheduler_thread.start()
        
        logger.info("Scheduler started")
        return self.running
    
    def stop(self):
        """Stop the scheduler and clean up all resources"""
        if not self.running:
            return
            
        logger.info("Initiating scheduler shutdown")
        self.running = False

        # Close job queue first to prevent new jobs
        try:
            self.job_queue.close()
            self.job_queue.cancel_join_thread()
        except Exception as e:
            logger.error(f"Error closing job queue: {e}")

        # Delegate worker pool shutdown
        try:
            self.worker_pool.shutdown()
        except Exception as e:
            logger.error(f"Error shutting down worker pool: {e}")
        
        # Wait for scheduler loop to exit
        if self.scheduler_thread and self.scheduler_thread.is_alive():
            try:
                logger.info("Waiting for scheduler thread to terminate")
                self.scheduler_thread.join(timeout=10)
                if self.scheduler_thread.is_alive():
                    logger.warning("Scheduler thread did not terminate within timeout")
            except Exception as e:
                logger.error(f"Error stopping scheduler thread: {e}")

        # Clear references to managed objects
        self.shared = None
        self.locks = None

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

    def _calculate_submission_delay(self):
        """Calculate delay to prevent thundering herd effect"""
        base_delay = 0.5
        
        if len(self.job_completion_times) > 5:
            avg_completion_time = sum(self.job_completion_times) / len(self.job_completion_times)
            scale_factor = min(avg_completion_time / 60.0, 3.0)
        else:
            scale_factor = 1.0
        
        scaled_delay = base_delay * scale_factor
        jitter = random.uniform(-0.3, 0.3) * scaled_delay
        final_delay = max(0.1, scaled_delay + jitter)
        
        return final_delay

    def _handle_job_completion(self, result):
        """Handle job completion - only release if resources were actually allocated"""
        job = result['job']
        device = result['device']
        resources_allocated = result.get('resources_allocated', False)
        
        # Track completion time
        if 'start_time' in result:
            completion_time = result['stop_time'] - result['start_time']
            self.job_completion_times.append(completion_time)
            if len(self.job_completion_times) > self.max_completion_history:
                self.job_completion_times.pop(0)
        
        # Only release resources if they were actually allocated
        if resources_allocated:
            if device and device.type == 'cuda':
                gpu_id = device.index 
                self.resource_manager.release_gpu(gpu_id)
            self.resource_manager.release_cpu()
        else:
            logger.debug(f"Skipping resource release for job {job.i}-{job.j} - no resources allocated")
        
        # Handle errors
        if result['status'] == 'error':
            if result.get('error_type') == 'OOM':
                logger.error(f"OOM job {job.i}-{job.j} on device {device}")
                if resources_allocated and device:
                    self.resource_manager.handle_oom(device)
            else:
                logger.error(f"Unknown error on job {job.i}-{job.j} on device {device}")
        else:
            logger.info(f"Completed job {job.i}-{job.j} on device {device}")

    def _schedule_jobs(self, n_jobs=1):
        if not self.resource_manager.cpu_manager.can_allocate_job():
            logger.debug("No resources available - stopping job scheduling")
        else:
            for _ in range(n_jobs):
                try:
                    job = self.job_queue.get(timeout=0.1)
                    if self._try_schedule_job(job):
                        logger.info(f"Scheduled job {job.i}-{job.j}")
                    else:
                        # Put back in queue
                        logger.debug(f"Failed to schedule job {job.i}-{job.j}")
                        self.job_queue.put(job)
                        time.sleep(1)  # Wait before retry
                except queue.Empty:
                    pass

    def _try_schedule_job(self, job):
        """Try to schedule a single job, returns True if scheduled, False otherwise"""
        device = self.resource_manager.try_allocate_resources()

        if device is not None:
            try:
                logger.debug(f"Scheduled job on device {device}")
                self.worker_pool.submit_job(job, device, self.resource_manager.cpu_manager.cores_per_job)
                
                # Thundering herd prevention
                delay = self._calculate_submission_delay()
                time.sleep(delay)
                return True
                
            except Exception as e:
                # Job submission failed - must release resources
                logger.error(f"Failed to submit job, releasing resources: {e}")
                if device.type == 'cuda':
                    self.resource_manager.release_gpu(device.index)
                self.resource_manager.release_cpu()
                return False
        return False

    def _adjust_concurrency(self):
        """Adjust concurrency based on completion rate and system resources"""
        completed_count = self.completed_count
        jobs_completed_in_interval = completed_count - self.last_completed_count
        cpu_usage = self.resource_manager.get_cpu_stats()

        if completed_count % 10 == 0 and completed_count > 0:
            logger.info(f"Progress: {completed_count}/{self.total_queued_jobs} jobs completed")

        # Decrease if CPU is overloaded
        if cpu_usage['cpu_percent'] > 80 and self.target_concurrency > 1:
            self.target_concurrency = max(1, self.target_concurrency - 1)
            self.job_buffer = int(round(self.target_concurrency / 2 + 0.5)) + 1
            logger.info(f"Decreased concurrency to {self.target_concurrency} due to high CPU usage")
        
        # Only increase if:
        # 1. Jobs are completing (not stuck)
        # 2. We're near our concurrency target (system can handle current load)
        # 3. We have available resources
        elif (jobs_completed_in_interval > 0 and 
            self.jobs_in_flight >= self.target_concurrency * 0.8 and 
            self.resource_manager.cpu_manager.can_allocate_job()):
            if self.target_concurrency < self.worker_pool.max_workers:
                self.target_concurrency += 1
                logger.info(f"Increased concurrency to {self.target_concurrency} "
                            f"({jobs_completed_in_interval} jobs completed in last {self.scaling_interval}s)")
        
        # Log warning if no progress
        elif jobs_completed_in_interval == 0:
            logger.warning(f"No jobs completed in last {self.scaling_interval}s - "
                            f"{self.jobs_in_flight} jobs in flight, concurrency: {self.target_concurrency}")

        # adjust GPU concurrency
        if self.resource_manager.gpu_manager:
            self.resource_manager.gpu_manager.adjust_concurrency()
        
        # propagate changes to workers
        self.worker_pool.scale_workers_to(self.target_concurrency)
        
        if self.resource_manager.gpu_manager:
            logger.info(self.resource_manager.gpu_manager.get_status())
        logger.info(self.resource_manager.cpu_manager.get_status())
        return completed_count

    def _scheduler_loop(self):
        """Main scheduler loop"""
        last_concurrency_check = time.time()
        logger.info(f"Jobs in queue: {self.jobs_in_queue}")
        logger.info(f"Jobs in flight: {self.jobs_in_flight}")
        
        try:
            while self.running:
                # Schedule new jobs
                jobs_in_worker_backlog, target_concurrency, job_buffer, jobs_in_flight = self.jobs_in_worker_backlog, self.target_concurrency, self.job_buffer, self.jobs_in_flight
                n_jobs = target_concurrency + job_buffer - jobs_in_flight - jobs_in_worker_backlog
                logger.debug(f'Schedule more? target: {target_concurrency}, buffer: {job_buffer}, jobs-in-flight: {jobs_in_flight}, jobs-in-worker-backlog {jobs_in_worker_backlog}' + (f'-> adding {n_jobs}' if n_jobs > 0 else f'-> no action {n_jobs}'))
                if n_jobs > 0 and self.running:
                    self._schedule_jobs(n_jobs=n_jobs)

                while self.running:
                    result = self.worker_pool.get_result(timeout=0.1)
                    if result is None or not self.running:
                        break
                    self._handle_job_completion(result)

                # Adjust concurrency based on completion rate
                current_time = time.time()
                if (current_time - last_concurrency_check >= self.scaling_interval) and self.running:
                    self.last_completed_count = self._adjust_concurrency()
                    last_concurrency_check = current_time
                    avg_completion_time = round(sum(self.job_completion_times) / len(self.job_completion_times))
                    self.scaling_interval = min(max(10, avg_completion_time), 180)

                if (self.jobs_in_queue == 0) and (self.jobs_in_flight == 0):
                    self.running = False
                else:
                    time.sleep(self.scheduler_loop_delay)

        except Exception as e:
            logger.exception(f"Scheduler loop error: {e}")
        finally:
            logger.info("Scheduler loop exiting")
            self.running = False
            sys.exit(0)

def generic_parallel_grid_search(
    # Core parameters
    job_factory: Callable,
    total_configs: int,
    samples_per_config: int,
    output_path: Path,
    
    # Resource parameters
    gpu_memory_per_job_gb: float = None,
    cpu_memory_per_job_gb: float = None,
    cpu_cores_per_job: int = 1,
    
    # Optional callbacks
    save_config: Callable[[Path], None] = None,
    process_results: Callable[[List[Dict], Dict, Path], Any] = None,
    ) -> Tuple[List[Dict], Dict]:
    """
    Generic parallel grid search that works with any job type.
    
    Args:
        job_factory: Function that creates jobs (passed to GenericJobGenerator)
        total_configs: Number of configurations to test
        samples_per_config: Number of samples per configuration
        output_path: Path to save results
        gpu_memory_per_job_gb: GPU memory per job
        cpu_memory_per_job_gb: CPU memory per job
        cpu_cores_per_job: CPU cores per job
        save_config: Optional callback to save configuration
        process_results: Optional callback to process results
        
    Returns:
        Tuple of (history, best_params)
    """
    
    # Create output directory
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=False)
    
    # Save configuration if callback provided
    if save_config:
        save_config(output_path)
    
    # Initialize scheduler
    resource_manager = ComputeJobResourceManager(
        cpu_memory_per_job_gb=cpu_memory_per_job_gb,
        cpu_cores_per_job=cpu_cores_per_job,
        gpu_memory_per_job_gb=gpu_memory_per_job_gb,
    )
    
    # Create job generator
    job_generator = GenericJobGenerator(
        job_factory=job_factory,
        total_configs=total_configs,
        samples_per_config=samples_per_config
    )
    
    try:
        with ResourceAwareScheduler(resource_manager=resource_manager) as scheduler:
            total_jobs = len(job_generator)
            logger.info(f"Submitting {total_jobs} jobs to scheduler")
            
            for job in job_generator:
                scheduler.add_job(job)
            
            scheduler.start()
            
            # Wait for completion
            start_time = time.time()
            with tqdm(total=total_jobs, desc="Grid Search Progress", unit="job") as pbar:
                last_count = 0
                while scheduler.completed_count < total_jobs and scheduler.running:
                    current_count = scheduler.completed_count
                    if current_count > last_count:
                        pbar.update(current_count - last_count)
                        last_count = current_count
                    time.sleep(0.5)
            
            if scheduler.completed_count == total_jobs:
                elapsed_time = time.time() - start_time
                logger.info(f"Grid search completed in {elapsed_time:.1f}s")
                
                # Extract results from shared state
                history = None
                best_params = None

                shared = job_generator.shared 
                locks = job_generator.locks
                with locks.get('history', locks):
                    history = list(shared.get('history', []))
                
                with locks.get('best_params', locks):
                    best_params = dict(shared.get('best_params', {}))
                
                # Process results if callback provided
                if process_results:
                    process_results(history, best_params, output_path)
                else:
                    # Default: save history as HDF5
                    if history:
                        history_df = pd.DataFrame(history)
                        file_path = output_path / 'log.h5'
                        history_df.to_hdf(file_path, key='df', mode='w')
                        logger.info(f"Saved {len(history)} results to {file_path}")
                
                return history, best_params
                
    except KeyboardInterrupt:
        logger.info("Main function received KeyboardInterrupt, ensuring cleanup")
        raise
    except Exception as e:
        logger.exception(f"Unexpected error in grid search: {e}")
        raise