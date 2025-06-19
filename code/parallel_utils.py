import time
import psutil
import pynvml
import torch
from collections import defaultdict, deque
from os import environ, sched_getaffinity, cpu_count
import torch.multiprocessing as mp
from abc import ABC, abstractmethod
from typing import Dict, Any
import numpy as np

import logging
logger = logging.getLogger(__name__)

class JobInterface(ABC):
    """Simple interface that all jobs must implement"""
    
    def __init__(self, i: int, j: int, total_configs: int, total_samples: int, shared: dict, locks: dict):
        self.i = i
        self.j = j
        self.total_configs = total_configs
        self.total_samples = total_samples
        self.shared = shared
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
        except torch.cuda.OutOfMemoryError as e:
            logger.error(f"GPU OOM for job {self.i}-{self.j}")
            if 'cuda' in str(device):
                with torch.cuda.device(device):
                    torch.cuda.empty_cache()
            return {'status': 'error', 'error_type': 'OOM', 'error': str(e)}
        except Exception as e:
            logger.exception(f"Error in job {self.i}-{self.j}: {e}")
            return {'status': 'error', 'error_type': 'general', 'error': str(e)}
    

    def get_log_prefix(self):
        """Generate log prefix with config/sample info"""
        config_str = f"{self.i+1:0{self._config_pad}d}/{self.total_configs}"
        sample_str = f"{self.j+1:0{self._sample_pad}d}/{self.total_samples}"
        return f"Config: {config_str}, Sample: {sample_str}"


class GenericJobGenerator:
    """Generic job generator that can create any type of job"""
    
    def __init__(self, job_factory, total_configs: int, samples_per_config: int):
        """
        job_factory: A callable that takes (i, j, total_configs, total_samples, shared, locks) and returns a job
        """
        self.job_factory = job_factory
        self.total_configs = total_configs
        self.samples_per_config = samples_per_config

        mp_manager = mp.Manager()
        self.shared = {
            'best_params': mp_manager.dict({'loss': None, 'params': None}),
            'dataset_cache': mp_manager.dict(),
            'dataset_scores': mp_manager.dict(),
            'history': mp_manager.list(),
        }
        self.locks = {
            'best_params': mp_manager.Lock(),
            'dataset_cache': mp_manager.Lock(),
            'dataset_scores': mp_manager.Lock(),
            'history': mp_manager.Lock(),
        }
    
    def __iter__(self):
        for i in range(self.total_configs):
            for j in range(self.samples_per_config):
                yield self.job_factory(
                    i=i,
                    j=j,
                    total_configs=self.total_configs,
                    total_samples=self.samples_per_config,
                    shared=self.shared,
                    locks=self.locks
                )
    
    def __len__(self):
        return self.total_configs * self.samples_per_config


def grouped_bar_graph(values, width=32):
    """Create a compact bar graph"""
    def bar_graph(values):
        # Unicode blocks for 8 levels
        bars = "▁▂▃▄▅▆▇█"
        return ''.join(bars[min(int(p / 100 * (len(bars) - 1)), len(bars) - 1)] for p in values)

    values = np.array(values)
    n = len(values)
    if n <= width:
        return bar_graph(values)
    else:
        # Average values over chunks
        chunks = np.array_split(values, width)
        avg_per_chunk = [chunk.mean() for chunk in chunks]
        return bar_graph(avg_per_chunk)


class CPUJobResourceManager:
    """Simplified CPU resource manager with SLURM support"""
    def __init__(self, memory_per_job_gb=2.0, cores_per_job=1, 
                 max_memory_usage=0.8, max_cpu_usage=0.8, max_jobs=None):
        self.memory_per_job_gb = memory_per_job_gb
        self.cores_per_job = cores_per_job
        self.max_memory_usage = max_memory_usage
        self.max_cpu_usage = max_cpu_usage
        
        # Simple state
        self.allocated_jobs = 0
        
        # Detect environment constraints
        self._detect_constraints()
        
        # Calculate max concurrent jobs
        self.max_jobs = max_jobs if max_jobs else self._calculate_max_concurrent()
        
        self._log_initialization()
    
    @staticmethod
    def _get_affinity_based_cpu_cores():
        """Get CPU cores available to this process"""
        try:
            return sched_getaffinity(0)
        except (AttributeError, OSError):
            # Not available on all systems (e.g., macOS)
            return None

    @staticmethod
    def _safe_get_core_count():
        """Get available CPU count, return (is_slurm, cpu_count)"""
        # CPU detection
        if 'SLURM_CPUS_PER_TASK' in environ:
            available_cpus = int(environ['SLURM_CPUS_PER_TASK'])
            is_slurm = True
        else:
            affinity_cores = CPUJobResourceManager._get_affinity_based_cpu_cores()
            if affinity_cores is not None:
                available_cpus = len(affinity_cores)
            else:
                # Fallback to total CPU count
                available_cpus = cpu_count() or 1
            is_slurm = False
        return is_slurm, available_cpus

    def _detect_constraints(self):
        """Detect SLURM or other environment constraints"""
        # CPU detection
        self.is_slurm, self.available_cpus = CPUJobResourceManager._safe_get_core_count()
        
        # Memory detection
        if 'SLURM_MEM_PER_NODE' in environ:
            # SLURM memory is usually in MB
            self.available_memory_gb = int(environ['SLURM_MEM_PER_NODE']) / 1024
        elif 'SLURM_MEM_PER_CPU' in environ:
            # Memory per CPU in MB
            mem_per_cpu_mb = int(environ['SLURM_MEM_PER_CPU'])
            self.available_memory_gb = (mem_per_cpu_mb * self.available_cpus) / 1024
        else:
            # Use system memory
            self.available_memory_gb = psutil.virtual_memory().total / (1024**3)
    
    def _calculate_max_concurrent(self):
        """Calculate maximum concurrent jobs based on resources"""
        # Memory-based limit
        usable_memory = self.available_memory_gb * self.max_memory_usage
        memory_limited_jobs = int(usable_memory / self.memory_per_job_gb)
        
        # CPU-based limit
        usable_cpus = self.available_cpus * self.max_cpu_usage
        cpu_limited_jobs = int(usable_cpus / self.cores_per_job)
        
        # Take the minimum and ensure at least 1
        return max(1, min(memory_limited_jobs, cpu_limited_jobs))
    
    def _log_initialization(self):
        """Log initialization details"""
        logger.info(f"CPU Resource Manager initialized:")
        logger.info(f"  Environment: {'SLURM' if self.is_slurm else 'Standard'}")
        logger.info(f"  Available CPUs: {self.available_cpus}")
        logger.info(f"  Available Memory: {self.available_memory_gb:.1f}GB")
        logger.info(f"  Memory per job: {self.memory_per_job_gb}GB")
        logger.info(f"  CPU cores per job: {self.cores_per_job}")
        logger.info(f"  Max concurrent jobs: {self.max_jobs}")
    
    def _get_allocated_cpu_cores(self):
        """Get the set of CPU core IDs available to this process"""
        if self.is_slurm:
            # In SLURM, we might have specific cores allocated
            return set(range(self.available_cpus))
        
        affinity_cores = CPUJobResourceManager._get_affinity_based_cpu_cores()
        if affinity_cores is not None:
            return affinity_cores
        else:
            return set(range(self.available_cpus))
    
    def get_system_usage(self):
        """Get current resource usage including per-core CPU"""
        # Memory usage
        mem = psutil.virtual_memory()
        memory_percent = mem.percent
        memory_available_gb = mem.available / (1024**3)
        
        # Get per-core CPU usage for our allocated cores
        try:
            all_cpu_usage = psutil.cpu_percent(interval=0.1, percpu=True)
            available_cores = self._get_allocated_cpu_cores()
            
            # Filter to only our allocated cores
            cpu_per_core = [all_cpu_usage[i] for i in available_cores if i < len(all_cpu_usage)]
            # Fixed: Handle empty list case
            cpu_percent = sum(cpu_per_core) / len(cpu_per_core) if cpu_per_core else 0
        except Exception:
            # Fallback to simple average
            cpu_percent = psutil.cpu_percent(interval=0.1)
            cpu_per_core = None
        
        return {
            'memory_percent': memory_percent,
            'memory_available_gb': memory_available_gb,
            'cpu_percent': cpu_percent,
            'cpu_per_core': cpu_per_core,
            'swap_percent': psutil.swap_memory().percent,
        }
    
   
    def can_allocate_job(self):
        """Check if resources are available for a new job"""
        # First check against max concurrent limit
        if self.allocated_jobs >= self.max_jobs:
            return False
        
        # Get current usage
        usage = self.get_system_usage()
        
        # Check memory availability
        if usage['memory_available_gb'] < self.memory_per_job_gb * 1.1:  # 10% buffer
            logger.debug(f"Insufficient memory: {usage['memory_available_gb']:.1f}GB available")
            return False
        
        # Check if we have per-core CPU data
        cpu_per_core = usage.get('cpu_per_core')
        if cpu_per_core and self.available_cpus > 0:  # Fixed: Check for empty list
            # Check if at least cores_per_job of our allocated cores are idle
            idle_cores = sum(core < 10.0 for core in cpu_per_core)  # < 10% usage = idle
            idle_ratio = idle_cores / self.available_cpus 
            
            logger.debug(f"Status CPU cores - Idle cores: {idle_cores}/{self.available_cpus} ({idle_ratio:.1%}) - {grouped_bar_graph(cpu_per_core)}")

            required_ratio = self.cores_per_job/self.available_cpus
            if idle_ratio < required_ratio:
                logger.debug(f"Not enough idle cores: {idle_ratio:.1%} < {required_ratio:.1%}")
                return False
        else:
            # Fallback to average CPU usage
            if usage['cpu_percent'] > self.max_cpu_usage * 100:
                logger.debug(f"High CPU usage: {usage['cpu_percent']:.1f}%")
                return False

        if usage['swap_percent'] > 90:
            logger.debug(f"High swap: {usage['swap_percent']:.1%} > 90%")
            return False
        
        return True
    
    def try_allocate_cpu(self):
        """Try to allocate resources for a job"""
        if self.can_allocate_job():
            self.allocated_jobs += 1
            return True
        return False
    
    def release(self):
        """Release resources from a completed job"""
        self.allocated_jobs -= 1
    
    def handle_oom(self):
        """Handle out-of-memory error by adjusting limits"""
        # Reduce max concurrent jobs
        old_max = self.max_jobs
        self.max_jobs = max(1, int(self.max_jobs * 0.8))
        
        # Increase memory estimate
        self.memory_per_job_gb *= 1.2
        
        logger.warning(f"OOM detected: reduced max jobs {old_max} -> {self.max_jobs}, "
                          f"increased memory estimate to {self.memory_per_job_gb:.1f}GB")
    
    def get_status(self):
        """Get current status string"""
        usage = self.get_system_usage()
        env_type = "SLURM" if self.is_slurm else "System"
        
        status = (f"{env_type}: {self.allocated_jobs}/{self.max_jobs} jobs | "
                  f"{usage['memory_available_gb']:.1f}GB free | "
                  f"{usage['memory_percent']:.0f}% mem | "
                  f"{usage['cpu_percent']:.0f}% cpu")
        
        # Add CPU bar graph if available
        if usage.get('cpu_per_core') and len(usage['cpu_per_core']) > 0:
            status += f" | cores: {grouped_bar_graph(usage['cpu_per_core'])}"
        
        return status
        
    def total_cpu_cores(self):
        """Get total available CPU cores"""
        return self.available_cpus  # Fixed: Return just the count, not tuple
    

class GPUJobResourceManager:
    def __init__(self, memory_per_job_gb=2.0, buffer_mb=100, memory_reset_threshold=0.9, initial_concurrency=2):
        self.memory_per_job_gb = memory_per_job_gb
        self.buffer_mb = buffer_mb
        self.memory_reset_threshold = memory_reset_threshold
        self.initial_concurrency = initial_concurrency
        
        # Simple counters (no multiprocessing)
        self.gpu_allocated_jobs = {}
        self.gpu_max_concurrent = {}
        self.gpu_reset_pending = {}
        self.last_allocated_gpu = -1
        
        # Unified metrics tracking
        self.gpu_metrics = defaultdict(lambda: {
            'memory_util': {'history': deque(maxlen=10), 'ewma': 0.0},
            'compute_util': {'history': deque(maxlen=10), 'ewma': 0.0},
            'last_sample_time': {'memory': 0, 'compute': 0}
        })
        self.sample_interval = 1.0
        self.ewma_alpha = 0.3  # Exponential weighted moving average factor

        # Track utilization by concurrency level
        self.concurrency_performance = defaultdict(lambda: defaultdict(lambda: {'ewma': 0.0}))
        
        self._init_slurm_gpus()
        self._init_pynvml()

    def _init_slurm_gpus(self):
        """Initialize only GPUs allocated by SLURM"""
        self.available_gpus = []
        
        # Check SLURM GPU allocation
        slurm_gpus = environ.get('CUDA_VISIBLE_DEVICES')
        if slurm_gpus:
            try:
                self.available_gpus = [int(gpu.strip()) for gpu in slurm_gpus.split(',')]
                logger.info(f"SLURM allocated GPUs: {self.available_gpus}")
            except ValueError:
                logger.error(f"Invalid CUDA_VISIBLE_DEVICES format: {slurm_gpus}")
                raise RuntimeError(f"Cannot parse CUDA_VISIBLE_DEVICES: {slurm_gpus}")
        
        # Fallback: use all available GPUs if no SLURM allocation
        if not self.available_gpus:
            if torch.cuda.is_available():
                self.available_gpus = list(range(torch.cuda.device_count()))
                logger.info(f"No SLURM GPU allocation found. Using all GPUs: {self.available_gpus}")
            else:
                raise RuntimeError("No GPUs available and CUDA not available")

    def _init_pynvml(self):
        """Initialize NVIDIA ML library and GPU tracking"""
        try:
            pynvml.nvmlInit()
        except Exception as e:
            logger.error(f"Failed to initialize pynvml: {e}")
            raise RuntimeError("pynvml initialization failed")
        
        failed_gpus = []
        
        # Initialize tracking for available GPUs only
        for gpu_id in self.available_gpus:
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
                device_name = pynvml.nvmlDeviceGetName(handle)
                if isinstance(device_name, bytes):
                    device_name = device_name.decode('utf-8')
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                
                logger.info(f"GPU {gpu_id}: {device_name}, "
                               f"Total Memory: {mem_info.total/(1024**3):.2f}GB")
                
                self.gpu_allocated_jobs[gpu_id] = 0
                self.gpu_max_concurrent[gpu_id] = self.initial_concurrency
                self.gpu_reset_pending[gpu_id] = False
                
            except Exception as e:
                logger.warning(f"Failed to initialize GPU {gpu_id}: {e}")
                failed_gpus.append(gpu_id)
        
        # Remove failed GPUs after iteration
        for gpu_id in failed_gpus:
            self.available_gpus.remove(gpu_id)
        
        if not self.gpu_allocated_jobs:
            raise RuntimeError("No GPUs could be initialized")

    def _update_metric(self, gpu_id, metric_type, value, force_update=False):
        """Update metric with history tracking and EWMA calculation"""
        current_time = time.time()
        metrics = self.gpu_metrics[gpu_id]
        
        # Check if we should update based on sample interval
        last_time_key = 'memory' if metric_type == 'memory_util' else 'compute'
        should_update = (current_time - metrics['last_sample_time'][last_time_key] >= self.sample_interval) or force_update
        
        if should_update:
            # Update history
            metrics[metric_type]['history'].append(value)
            metrics['last_sample_time'][last_time_key] = current_time
            
            # Update EWMA
            if metrics[metric_type]['ewma'] == 0.0:
                # Initialize EWMA with first value
                metrics[metric_type]['ewma'] = value
            else:
                metrics[metric_type]['ewma'] = (self.ewma_alpha * value + 
                                               (1 - self.ewma_alpha) * metrics[metric_type]['ewma'])
        
        return metrics[metric_type]['ewma']

    def _get_metric_average(self, gpu_id, metric_type):
        """Get average value for a metric (uses EWMA)"""
        return self.gpu_metrics[gpu_id][metric_type]['ewma']

    def _get_gpu_memory_info(self, gpu_id):
        """Get current GPU memory usage with unified tracking"""
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            
            memory_utilization = mem_info.used / mem_info.total if mem_info.total > 0 else 0
            
            # Update memory utilization using unified method
            avg_memory_util = self._update_metric(gpu_id, 'memory_util', memory_utilization)
            
            # Trigger reset if threshold exceeded
            if avg_memory_util > self.memory_reset_threshold:
                self._mark_for_reset(gpu_id, avg_memory_util)
            
            return mem_info.total, mem_info.used, mem_info.free, avg_memory_util
            
        except Exception as e:
            logger.error(f"Error getting GPU {gpu_id} memory info: {e}")
            return 0, 0, 0, 0

    def _get_gpu_compute_utilization(self, gpu_id, samples=3):
        """Get GPU compute utilization with unified tracking"""
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
            util_samples = []
            
            # Take multiple samples for stability
            for _ in range(samples):
                util_samples.append(pynvml.nvmlDeviceGetUtilizationRates(handle).gpu)
                if samples > 1:
                    time.sleep(1.0 / samples)
            
            compute_utilization = sum(util_samples) / len(util_samples) / 100.0  # Convert to 0-1 range
            
            # Update compute utilization using unified method
            avg_compute_util = self._update_metric(gpu_id, 'compute_util', compute_utilization, force_update=True)
            
            # Update concurrency performance tracking
            concurrency_level = self.gpu_max_concurrent[gpu_id]
            self._update_concurrency_performance(gpu_id, concurrency_level, compute_utilization)
            
            return compute_utilization, avg_compute_util
            
        except Exception as e:
            logger.warning(f"Failed to get GPU {gpu_id} compute utilization: {e}")
            return 1.0, 1.0  # Assume high utilization on error

    def _update_concurrency_performance(self, gpu_id, concurrency_level, utilization):
        """Track performance at different concurrency levels"""
        perf = self.concurrency_performance[gpu_id][concurrency_level]
        if perf['ewma'] == 0.0:
            perf['ewma'] = utilization
        else:
            perf['ewma'] = self.ewma_alpha * utilization + (1 - self.ewma_alpha) * perf['ewma']

    def _mark_for_reset(self, gpu_id, memory_utilization):
        """Mark GPU for reset when memory usage is too high"""
        if not self.gpu_reset_pending[gpu_id]:
            self.gpu_reset_pending[gpu_id] = True
            logger.warning(f"GPU {gpu_id} memory at {memory_utilization:.1%} - "
                              f"marked for reset after current jobs complete")

    def can_allocate_job_on_this_gpu(self, gpu_id):
        """Check if we can allocate a job to this GPU"""
        if gpu_id not in self.available_gpus or self.gpu_reset_pending[gpu_id]:
            return False
        
        try:
            mem_total, mem_used, mem_free, avg_memory_util = self._get_gpu_memory_info(gpu_id)
            required_memory_bytes = (self.memory_per_job_gb * 1024**3) + (self.buffer_mb * 1024**2)
            
            # Don't allocate if average usage too high
            if avg_memory_util > 0.85:
                return False
                
            return mem_free >= required_memory_bytes
            
        except Exception as e:
            logger.error(f"Error checking GPU {gpu_id}: {e}")
            return False

    def should_increase_concurrency(self, gpu_id):
        """Check if we should increase concurrency on this GPU"""
        if gpu_id not in self.available_gpus or self.has_pending_resets():
            return False
        
        if self.gpu_allocated_jobs[gpu_id] < self.gpu_max_concurrent[gpu_id]:
            return False

        # Check memory availability
        try:
            mem_total, mem_used, mem_free, avg_memory_util = self._get_gpu_memory_info(gpu_id)
            required_memory_bytes = (self.memory_per_job_gb * 1024**3) + (self.buffer_mb * 1024**2)
            
            if avg_memory_util > 0.85 or mem_free < required_memory_bytes:
                return False
                
        except Exception as e:
            logger.error(f"Error checking GPU {gpu_id} memory: {e}")
            return False

        # Check compute utilization
        _, avg_compute_util = self._get_gpu_compute_utilization(gpu_id)
        
        # Only increase if compute utilization is moderate (not too low, not too high)
        return 0.2 < avg_compute_util <= 0.85

    def should_decrease_concurrency(self, gpu_id):
        """Check if we should decrease concurrency on this GPU"""
        if gpu_id not in self.available_gpus:
            return False
            
        perf_data = self.concurrency_performance[gpu_id]
        if len(perf_data) < 2:
            return False
            
        current_level = self.gpu_max_concurrent[gpu_id]
        
        # Find best performing concurrency level
        best_level = max(perf_data.keys(), key=lambda l: perf_data[l]['ewma'])
        
        # Decrease if current level performs worse than best level
        return current_level > best_level and current_level > 1
    
    def can_allocate_job(self):
        """Check if a GPU is available for allocation (does not mutate state)."""
        return self._get_next_available_gpu(allocate=False) is not None

    def try_allocate_gpu(self):
        """Try to allocate a GPU for a new job (mutates state)."""
        gpu_id = self._get_next_available_gpu(allocate=True)
        if gpu_id is None:
            logger.debug("No GPU could be allocated (all at capacity or pending reset)")
        return gpu_id
    
    def _get_next_available_gpu(self, allocate: bool = False):
        """Shared logic for GPU selection.

        Args:
            allocate (bool): If True, will increment job counter and update allocation state.
        
        Returns:
            gpu_id or None
        """
        if not self.available_gpus:
            return None

        gpu_count = len(self.available_gpus)
        start_index = self.last_allocated_gpu

        for i in range(gpu_count):
            index = (start_index + 1 + i) % gpu_count
            gpu_id = self.available_gpus[index]

            if self.gpu_reset_pending[gpu_id]:
                continue

            current_jobs = self.gpu_allocated_jobs[gpu_id]
            max_jobs = self.gpu_max_concurrent[gpu_id]

            if current_jobs < max_jobs and self.can_allocate_job_on_this_gpu(gpu_id):
                if allocate:
                    self.last_allocated_gpu = index
                    self.gpu_allocated_jobs[gpu_id] += 1
                    logger.debug(f"Allocated GPU {gpu_id} "
                                    f"({self.gpu_allocated_jobs[gpu_id]}/{max_jobs} jobs)")
                return gpu_id

        return None


    def release(self, gpu_id):
        """Release GPU and perform reset if needed"""
        if gpu_id in self.gpu_allocated_jobs:
            self.gpu_allocated_jobs[gpu_id] -= 1 
            
            # Perform reset if pending and no jobs remain
            if self.gpu_reset_pending[gpu_id] and self.gpu_allocated_jobs[gpu_id] == 0:
                self._perform_reset(gpu_id)

    def _perform_reset(self, gpu_id):
        """Clear GPU memory and reset state"""
        try:
            logger.info(f"Resetting GPU {gpu_id}")
            
            # Clear GPU memory
            with torch.cuda.device(gpu_id):
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            # Clear metrics history
            metrics = self.gpu_metrics[gpu_id]
            metrics['memory_util']['history'].clear()
            metrics['memory_util']['ewma'] = 0.0
            metrics['compute_util']['history'].clear()
            metrics['compute_util']['ewma'] = 0.0
            
            # Reset flags
            self.gpu_reset_pending[gpu_id] = False
            
            logger.info(f"GPU {gpu_id} reset complete")
            
        except Exception as e:
            logger.error(f"Error resetting GPU {gpu_id}: {e}")
            self.gpu_reset_pending[gpu_id] = False

    def handle_oom(self, gpu_id):
        """Handle out-of-memory error"""
        if gpu_id in self.gpu_allocated_jobs:
            # Trigger reset
            self._mark_for_reset(gpu_id, 1.0)
            
            # Immediate cache clear
            try:
                with torch.cuda.device(gpu_id):
                    torch.cuda.empty_cache()
            except Exception:
                pass

    def get_status(self):
        """Get current status of all GPUs"""
        status = []
        for gpu_id in self.available_gpus:
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                mem_free_gb = mem_info.free / (1024**3)
                mem_util_pct = (mem_info.used / mem_info.total) * 100
                
                jobs = self.gpu_allocated_jobs[gpu_id]
                max_jobs = self.gpu_max_concurrent[gpu_id]
                
                # Get average metrics
                avg_mem = self._get_metric_average(gpu_id, 'memory_util') * 100
                avg_compute = self._get_metric_average(gpu_id, 'compute_util') * 100
                
                status_str = f"GPU {gpu_id}: {jobs}/{max_jobs} jobs, " \
                           f"{mem_free_gb:.1f}GB free, {mem_util_pct:.0f}% mem used " \
                           f"(avg: {avg_mem:.0f}% mem, {avg_compute:.0f}% compute)"
                
                if self.gpu_reset_pending[gpu_id]:
                    status_str += " [RESET PENDING]"
                    
                status.append(status_str)
                
            except Exception:
                status.append(f"GPU {gpu_id}: unavailable")
        
        return " | ".join(status)

    def get_total_capacity(self):
        """Get total job capacity across all GPUs"""
        return sum(self.gpu_max_concurrent[gpu_id] for gpu_id in self.available_gpus)

    def has_pending_resets(self):
        """Check if any GPUs have pending resets"""
        return any(self.gpu_reset_pending[gpu_id] for gpu_id in self.available_gpus)

    def adjust_concurrency(self):
        for gpu_id in self.available_gpus:
            self._adjust_concurrency(gpu_id)

    def _adjust_concurrency(self, gpu_id):
        """Adjust concurrency based on performance metrics"""
        if gpu_id not in self.gpu_allocated_jobs or self.gpu_allocated_jobs[gpu_id] == 0:
            return False
            
        if self.should_increase_concurrency(gpu_id):
            self.gpu_max_concurrent[gpu_id] += 1
            logger.debug(f"Increased GPU {gpu_id} concurrency limit to {self.gpu_max_concurrent[gpu_id]}")
            logger.debug(self.get_status())
            return True
        elif self.should_decrease_concurrency(gpu_id):
            self.gpu_max_concurrent[gpu_id] = max(1, self.gpu_max_concurrent[gpu_id] - 1)
            logger.debug(f"Decreased GPU {gpu_id} concurrency limit to {self.gpu_max_concurrent[gpu_id]}")
            logger.debug(self.get_status())
            return True
            
        return False

    def get_metrics_summary(self, gpu_id):
        """Get a summary of metrics for a GPU"""
        if gpu_id not in self.gpu_metrics:
            return None
            
        metrics = self.gpu_metrics[gpu_id]
        return {
            'memory_util': {
                'current': metrics['memory_util']['history'][-1] if metrics['memory_util']['history'] else 0,
                'average': metrics['memory_util']['ewma'],
                'history': list(metrics['memory_util']['history'])
            },
            'compute_util': {
                'current': metrics['compute_util']['history'][-1] if metrics['compute_util']['history'] else 0,
                'average': metrics['compute_util']['ewma'],
                'history': list(metrics['compute_util']['history'])
            },
            'concurrency_performance': dict(self.concurrency_performance[gpu_id])
        }

class ComputeJobResourceManager:
    def __init__(self, cpu_memory_per_job_gb=2.0, cpu_cores_per_job=1, gpu_memory_per_job_gb=2.0):
        self.cpu_manager = CPUJobResourceManager(
            memory_per_job_gb=cpu_memory_per_job_gb,
            cores_per_job=cpu_cores_per_job,
            max_memory_usage=0.8,
            max_cpu_usage=0.8
        )
        
        self.use_gpu = torch.cuda.is_available()
        self.gpu_manager = None
        if self.use_gpu:
            self.gpu_manager = GPUJobResourceManager(memory_per_job_gb=gpu_memory_per_job_gb)
        logger.info(f"Mode: {'GPU+CPU' if self.use_gpu else 'CPU only'}")

    def try_allocate_resources(self):
        """Try to allocate resources for a job - atomic operation"""
        gpu_id = None
        cpu_allocated = False
        
        try:
            # Always try to allocate CPU first
            cpu_allocated = self.cpu_manager.try_allocate_cpu()
            if not cpu_allocated:
                return None
                
            # Try GPU if available
            if self.use_gpu:
                gpu_id = self.gpu_manager.try_allocate_gpu()
                if gpu_id is not None:
                    return torch.device(f'cuda:{gpu_id}')
            
            # CPU only mode or GPU allocation failed
            return torch.device('cpu')
            
        except Exception:
            # Clean up on any error
            if cpu_allocated:
                self.cpu_manager.release()
            if gpu_id is not None:
                self.gpu_manager.release(gpu_id)
            return None

    def release_gpu(self, gpu_id):
        if gpu_id is not None:
            return self.gpu_manager.release(gpu_id)
    
    def release_cpu(self):
        return self.cpu_manager.release()

    def handle_oom(self, device):
        if device.type == 'cuda':
            gpu_id = device.index
            self.gpu_manager.handle_oom(gpu_id)
        else:
            self.cpu_manager.handle_oom()

    def get_cpu_stats(self):
        return self.cpu_manager.get_system_usage()

    @property
    def max_concurrent(self):
        return self.cpu_manager.max_jobs
