import time
import psutil
import pynvml
import torch
from collections import defaultdict, deque
from os import environ, sched_getaffinity, cpu_count
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


class GenericJobGenerator:
    """Generic job generator that can create any type of job"""
    
    def __init__(self, manager, job_factory, total_configs: int, samples_per_config: int):
        """
        job_factory: A callable that takes (i, j, total_configs, total_samples, shared, locks) and returns a job
        """
        self.job_factory = job_factory
        self.total_configs = total_configs
        self.samples_per_config = samples_per_config

        self.mp_manager = manager 
        self.shared = {
            'history': self.mp_manager.list(),
        }
        self.locks = {
            'history': self.mp_manager.Lock(),
            'dataset_lock': self.mp_manager.Lock(), # in case there is data generation (write not just read)
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

# TODO in SLURM mode the effective cores are stuck at SLURM_CPUS_PER_TASK even if hypthread is True, fix atm is to run with priviledged and not specify so mode is SYSTEM
class CPUJobResourceManager:
    """CPU resource manager with SLURM support and hyperthreading awareness"""
    
    def __init__(self, memory_per_job_gb=2.0, cores_per_job=1, 
                 max_memory_usage=0.8, max_cpu_usage=0.9, max_jobs=None, 
                 allow_hyperthread=True, hyperthread_efficiency=0.9):
        self.memory_per_job_gb = memory_per_job_gb
        self.cores_per_job = cores_per_job
        self.max_memory_usage = max_memory_usage
        self.max_cpu_usage = max_cpu_usage
        self.allow_hyperthread = allow_hyperthread
        self.hyperthread_efficiency = hyperthread_efficiency
        
        self._detect_constraints()
        
        self.allocated_jobs = 0
        concurrency_estimate = self._calculate_max_concurrent_processes()
        self.target_concurrency = max(1, concurrency_estimate // 3)
        self.max_jobs = max_jobs if max_jobs else max(1, self.effective_cpus // cores_per_job)
        
        self._log_initialization()
    
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
        
        # Calculate effective CPUs (accounting for HT efficiency)
        if self.allow_hyperthread:
            ht_cores = max(0, self.logical_cpus - self.physical_cpus)
            base_effective = self.physical_cpus + int(ht_cores * self.hyperthread_efficiency)
        else:
            base_effective = self.physical_cpus
        
        # If SLURM, cap to allocation
        if self.slurm_cpus:
            self.effective_cpus = min(self.slurm_cpus, base_effective)
        else:
            self.effective_cpus = base_effective
        
        # available_cpus for legacy compatibility (used by _get_allocated_cpu_cores)
        self.available_cpus = self.slurm_cpus if self.slurm_cpus else self.logical_cpus
        
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

    def _calculate_max_concurrent_processes(self):
        """Calculate maximum concurrent jobs based on resources"""
        usable_memory = self.available_memory_gb * self.max_memory_usage
        memory_limited_jobs = int(usable_memory / self.memory_per_job_gb)
        
        # Leave 1 core for system, apply max_cpu_usage factor
        usable_cpus = max(0, (self.effective_cpus - 1) * self.max_cpu_usage)
        cpu_limited_jobs = int(usable_cpus / self.cores_per_job)
        
        return max(1, min(memory_limited_jobs, cpu_limited_jobs))
    
    def _log_initialization(self):
        """Log initialization details"""
        logger.info(f"CPU Resource Manager initialized:")
        logger.info(f"  Environment: {'SLURM' if self.is_slurm else 'Standard'}")
        if self.slurm_cpus:
            logger.info(f"  SLURM allocation: {self.slurm_cpus} CPUs")
        logger.info(f"  Physical cores: {self.physical_cpus}")
        logger.info(f"  Logical cores: {self.logical_cpus}")
        logger.info(f"  Effective capacity: {self.effective_cpus} (HT efficiency: {self.hyperthread_efficiency})")
        logger.info(f"  Available Memory: {self.available_memory_gb:.1f}GB")
        logger.info(f"  Memory per job: {self.memory_per_job_gb}GB")
        logger.info(f"  Cores per job: {self.cores_per_job}")
        logger.info(f"  Max concurrent jobs: {self.max_jobs}")
    
    def _get_allocated_cpu_cores(self):
        """Get the set of CPU core IDs available to this process"""
        affinity_cores = self._get_affinity_based_cpu_cores()
        if affinity_cores is not None:
            return affinity_cores
        # Fallback: assume we can use all logical cores up to our limit
        return set(range(self.available_cpus))

    def get_system_usage(self):
        """Get current resource usage"""
        mem = psutil.virtual_memory()
        
        try:
            all_cpu_usage = psutil.cpu_percent(interval=0.1, percpu=True)
            available_cores = self._get_allocated_cpu_cores()
            cpu_per_core = [all_cpu_usage[i] for i in available_cores if i < len(all_cpu_usage)]
            cpu_percent = sum(cpu_per_core) / len(cpu_per_core) if cpu_per_core else 0
        except Exception:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            cpu_per_core = None
        
        return {
            'memory_percent': mem.percent,
            'memory_available_gb': mem.available / (1024**3),
            'cpu_percent': cpu_percent,
            'cpu_per_core': cpu_per_core,
            'swap_percent': psutil.swap_memory().percent,
        }
   
    def has_available_cpu(self):
        """Check if resources are available for a new job"""
        if self.allocated_jobs >= self.max_jobs:
            logger.debug(f"At max jobs: {self.allocated_jobs}/{self.max_jobs}")
            return False
        
        usage = self.get_system_usage()
        
        # Memory check
        if usage['memory_available_gb'] < self.memory_per_job_gb * 1.1:
            logger.debug(f"Insufficient memory: {usage['memory_available_gb']:.1f}GB available")
            return False
        
        # CPU check - simple average-based (topology-agnostic)
        if usage['cpu_percent'] > self.max_cpu_usage * 100:
            logger.debug(f"CPU saturated: {usage['cpu_percent']:.1f}% > {self.max_cpu_usage * 100:.0f}%")
            return False
        
        # Swap pressure check
        if usage['swap_percent'] > 90:
            logger.debug(f"High swap: {usage['swap_percent']:.0f}%")
            return False
        
        return True
    
    def try_allocate_cpu(self):
        """Try to allocate resources for a job"""
        if self.has_available_cpu():
            self.allocated_jobs += 1
            return True
        return False
    
    def release(self):
        """Release resources from a completed job"""
        self.allocated_jobs = max(0, self.allocated_jobs - 1)
    
    def handle_oom(self):
        """Handle out-of-memory error by adjusting limits"""
        old_max = self.max_jobs
        self.max_jobs = max(1, int(self.max_jobs * 0.8))
        self.memory_per_job_gb *= 1.2
        
        logger.warning(f"OOM detected: reduced max jobs {old_max} -> {self.max_jobs}, "
                       f"increased memory estimate to {self.memory_per_job_gb:.1f}GB")
    
    def get_status(self):
        """Get current status string"""
        usage = self.get_system_usage()
        env_type = "SLURM" if self.is_slurm else "SYSTEM"
        
        status = (f"{env_type}: {self.allocated_jobs}/{self.max_jobs} jobs | "
                  f"{usage['memory_available_gb']:.1f}GB free | "
                  f"{usage['memory_percent']:.0f}% mem | "
                  f"{usage['cpu_percent']:.0f}% cpu")
        
        if usage.get('cpu_per_core'):
            status += f" | cores: {grouped_bar_graph(usage['cpu_per_core'])}"
        
        return status
    
    def total_cpu_cores(self):
        """Get total effective CPU capacity"""
        return self.effective_cpus

    def adjust_concurrency(self):
        if self.should_decrease_concurrency():
            self.target_concurrency = max(1, self.target_concurrency - 1)
        elif self.should_increase_concurrency():
            self.target_concurrency += 1
        return self.target_concurrency
    
    def should_decrease_concurrency(self):
        if self.target_concurrency <= 1:
            return False
        usage = self.get_system_usage()
        return usage['cpu_percent'] > self.max_cpu_usage * 100

    def should_increase_concurrency(self):
        return self.has_available_cpu() and (self.target_concurrency < self.max_jobs)


class GPUJobResourceManager:
    def __init__(self, memory_per_job_gb=2.0, buffer_gb=0.1, initial_concurrency=2):
        self.memory_per_job_gb = memory_per_job_gb
        self.buffer_gb = buffer_gb
        self.initial_concurrency = initial_concurrency
        
        self.gpu_allocated_jobs = {}
        self.gpu_target_concurrent = {}
        
        self.gpu_metrics = defaultdict(lambda: {
            'memory_util': {'history': deque(maxlen=10), 'ewma': 0.0},
            'compute_util': {'history': deque(maxlen=10), 'ewma': 0.0},
            'last_sample_time': {'memory': 0, 'compute': 0}
        })
        self.sample_interval = 1.0
        self.ewma_alpha = 0.3

        # Thresholds
        self.memory_util_threshold = 0.85
        self.compute_increase_threshold = 0.90   # Increase concurrency if below
        self.compute_saturated_threshold = 0.98  # Decrease concurrency if at/above
        
        self._init_slurm_gpus()
        self._init_pynvml()

    def _init_slurm_gpus(self):
        """Initialize only GPUs allocated by SLURM"""
        self.available_gpus = []
        
        slurm_gpus = environ.get('CUDA_VISIBLE_DEVICES')
        if slurm_gpus:
            try:
                self.available_gpus = [int(gpu.strip()) for gpu in slurm_gpus.split(',')]
                logger.info(f"SLURM allocated GPUs: {self.available_gpus}")
            except ValueError:
                raise RuntimeError(f"Cannot parse CUDA_VISIBLE_DEVICES: {slurm_gpus}")
        
        if not self.available_gpus:
            if torch.cuda.is_available():
                self.available_gpus = list(range(torch.cuda.device_count()))
                logger.info(f"No SLURM allocation. Using all GPUs: {self.available_gpus}")
            else:
                raise RuntimeError("No GPUs available and CUDA not available")

    def _init_pynvml(self):
        """Initialize NVIDIA ML library and GPU tracking"""
        try:
            pynvml.nvmlInit()
        except Exception as e:
            raise RuntimeError(f"pynvml initialization failed: {e}")
        
        failed_gpus = []
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
                self.gpu_target_concurrent[gpu_id] = self.initial_concurrency
            except Exception as e:
                logger.warning(f"Failed to initialize GPU {gpu_id}: {e}")
                failed_gpus.append(gpu_id)
        
        for gpu_id in failed_gpus:
            self.available_gpus.remove(gpu_id)
        
        if not self.gpu_allocated_jobs:
            raise RuntimeError("No GPUs could be initialized")

    def _update_metric(self, gpu_id, metric_type, value, force_update=False):
        """Update metric with history tracking and EWMA calculation"""
        metrics = self.gpu_metrics[gpu_id]
        last_time_key = 'memory' if metric_type == 'memory_util' else 'compute'
        current_time = time.time()
        
        if force_update or (current_time - metrics['last_sample_time'][last_time_key] >= self.sample_interval):
            metrics[metric_type]['history'].append(value)
            metrics['last_sample_time'][last_time_key] = current_time
            
            old_ewma = metrics[metric_type]['ewma']
            metrics[metric_type]['ewma'] = value if old_ewma == 0.0 else (
                self.ewma_alpha * value + (1 - self.ewma_alpha) * old_ewma
            )
        
        return metrics[metric_type]['ewma']

    def _get_gpu_memory_info(self, gpu_id):
        """Get current GPU memory usage with unified tracking"""
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            
            utilization = mem_info.used / mem_info.total if mem_info.total > 0 else 0
            avg_util = self._update_metric(gpu_id, 'memory_util', utilization)
            
            return mem_info.total, mem_info.used, mem_info.free, avg_util
        except Exception as e:
            logger.error(f"Error getting GPU {gpu_id} memory info: {e}")
            return 0, 0, 0, 0

    def _get_gpu_compute_utilization(self, gpu_id, samples=3):
        """Get GPU compute utilization with unified tracking"""
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
            util_samples = []
            
            for _ in range(samples):
                util_samples.append(pynvml.nvmlDeviceGetUtilizationRates(handle).gpu)
                if samples > 1:
                    time.sleep(1.0 / samples)
            
            utilization = sum(util_samples) / len(util_samples) / 100.0
            avg_util = self._update_metric(gpu_id, 'compute_util', utilization, force_update=True)
            
            return utilization, avg_util
        except Exception as e:
            logger.warning(f"Failed to get GPU {gpu_id} compute utilization: {e}")
            return 1.0, 1.0  # Assume saturated on error

    def _has_memory_capacity(self, gpu_id):
        """Check if GPU has enough memory for another job"""
        try:
            _, _, mem_free, avg_util = self._get_gpu_memory_info(gpu_id)
            required_bytes = (self.memory_per_job_gb + self.buffer_gb) * 1024**3
            return avg_util <= self.memory_util_threshold and mem_free >= required_bytes
        except Exception as e:
            logger.error(f"Error checking GPU {gpu_id} memory: {e}")
            return False

    def _can_allocate_on_gpu(self, gpu_id):
        """Check if we can allocate a job to this GPU"""
        return gpu_id in self.available_gpus and self._has_memory_capacity(gpu_id)

    def _should_increase_concurrency(self, gpu_id):
        """Check if we should increase concurrency on this GPU"""
        if gpu_id not in self.available_gpus:
            return False
        
        # Only raise ceiling if we've hit it
        if self.gpu_allocated_jobs[gpu_id] < self.gpu_target_concurrent[gpu_id]:
            return False

        if not self._has_memory_capacity(gpu_id):
            return False

        _, avg_compute = self._get_gpu_compute_utilization(gpu_id)
        return avg_compute < self.compute_increase_threshold

    def _should_decrease_concurrency(self, gpu_id):
        """Check if we should decrease concurrency on this GPU"""
        if gpu_id not in self.available_gpus:
            return False
        
        if self.gpu_target_concurrent[gpu_id] <= 1:
            return False
            
        _, avg_compute = self._get_gpu_compute_utilization(gpu_id)
        return avg_compute >= self.compute_saturated_threshold
    
    def has_available_gpu(self):
        """Check if a GPU is available for allocation (does not mutate state)"""
        return self._get_next_available_gpu(allocate=False) is not None

    def try_allocate_gpu(self):
        """Try to allocate a GPU for a new job (mutates state)"""
        gpu_id = self._get_next_available_gpu(allocate=True)
        if gpu_id is None:
            logger.debug("No GPU could be allocated (all at capacity)")
        return gpu_id
    
    def _get_next_available_gpu(self, allocate=False):
        if not self.available_gpus:
            return None
        
        best_gpu = None
        best_ratio = float('inf')
        
        for gpu_id in self.available_gpus:
            current = self.gpu_allocated_jobs[gpu_id]
            target = self.gpu_target_concurrent[gpu_id]
            if current < target and self._can_allocate_on_gpu(gpu_id):
                ratio = current / target
                if ratio < best_ratio:
                    best_ratio = ratio
                    best_gpu = gpu_id
        
        if best_gpu is not None and allocate:
            self.gpu_allocated_jobs[best_gpu] += 1
        
        return best_gpu

    def release(self, gpu_id):
        """Release GPU after job completion"""
        if gpu_id in self.gpu_allocated_jobs:
            self.gpu_allocated_jobs[gpu_id] = max(0, self.gpu_allocated_jobs[gpu_id] - 1)
            
    def handle_oom(self, gpu_id):
        """Handle out-of-memory error"""
        logger.warning(f"OOM on GPU {gpu_id}")
        
        if gpu_id in self.gpu_allocated_jobs:
            self.gpu_allocated_jobs[gpu_id] = 0
        
        # Clear stale metrics
        metrics = self.gpu_metrics[gpu_id]
        for key in ('memory_util', 'compute_util'):
            metrics[key]['history'].clear()
            metrics[key]['ewma'] = 0.0
        
        try:
            with torch.cuda.device(gpu_id):
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except Exception as e:
            logger.error(f"Failed to clear cache on GPU {gpu_id}: {e}")

    def get_status(self):
        """Get current status of all GPUs"""
        status = []
        for gpu_id in self.available_gpus:
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                
                jobs = self.gpu_allocated_jobs[gpu_id]
                target = self.gpu_target_concurrent[gpu_id]
                avg_mem = self.gpu_metrics[gpu_id]['memory_util']['ewma'] * 100
                avg_compute = self.gpu_metrics[gpu_id]['compute_util']['ewma'] * 100
                
                status.append(
                    f"GPU {gpu_id}: {jobs}/{target} jobs, "
                    f"{mem_info.free/(1024**3):.1f}GB free "
                    f"(avg: {avg_mem:.0f}% mem, {avg_compute:.0f}% compute)"
                )
            except Exception:
                status.append(f"GPU {gpu_id}: unavailable")
        
        return " | ".join(status)

    def get_total_capacity(self):
        """Get total job capacity across all GPUs"""
        return sum(self.gpu_target_concurrent[gpu_id] for gpu_id in self.available_gpus)

    def adjust_concurrency(self):
        """Adjust concurrency for all GPUs based on utilization"""
        for gpu_id in self.available_gpus:
            if self.gpu_allocated_jobs.get(gpu_id, 0) == 0:
                continue
                
            if self._should_increase_concurrency(gpu_id):
                self.gpu_target_concurrent[gpu_id] += 1
                logger.debug(f"Increased GPU {gpu_id} concurrency to {self.gpu_target_concurrent[gpu_id]}")
            elif self._should_decrease_concurrency(gpu_id):
                self.gpu_target_concurrent[gpu_id] -= 1
                logger.debug(f"Decreased GPU {gpu_id} concurrency to {self.gpu_target_concurrent[gpu_id]}")
        
        return self.gpu_target_concurrent

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
            }
        }
    

class ComputeJobResourceManager:
    def __init__(self, cpu_memory_per_job_gb=2.0, cpu_cores_per_job=1, gpu_memory_per_job_gb=2.0):
        self.cpu_manager = CPUJobResourceManager(
            memory_per_job_gb=cpu_memory_per_job_gb,
            cores_per_job=cpu_cores_per_job,
        )
        
        self.use_gpu = torch.cuda.is_available()
        self.gpu_manager = None
        if self.use_gpu:
            self.gpu_manager = GPUJobResourceManager(memory_per_job_gb=gpu_memory_per_job_gb)

        logger.info(f"Mode: {'GPU+CPU' if self.use_gpu else 'CPU only'}")

    def try_allocate_resources(self):
        """Try to allocate resources for a job - atomic operation"""
        allocated_device = None
        
        try:
            # Always try to allocate CPU first
            cpu_allocated = self.cpu_manager.try_allocate_cpu()
            if not cpu_allocated:
                return None
            
            # At this point, CPU is allocated
            allocated_device = torch.device('cpu')
                
            # Try GPU if available
            if self.use_gpu:
                gpu_id = self.gpu_manager.try_allocate_gpu()
                if gpu_id is not None:
                    allocated_device = torch.device(f'cuda:{gpu_id}')
            
            return allocated_device
            
        except Exception as e:
            # Clean up using the same release method
            if allocated_device is not None:
                self.release_resources(allocated_device)
            return None

    def release_resources(self, device):
        """Release resources based on device - single entry point"""
        if device.type == 'cuda':
            self.gpu_manager.release(device.index)
        self.cpu_manager.release()

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

    @property
    def target_concurrency_split_by_resource(self):
        targets = {torch.device('cpu'): self.cpu_manager.target_concurrency}
        if self.use_gpu:
            for gpu_id in self.gpu_manager.available_gpus:
                dev = torch.device(f'cuda:{gpu_id}')
                targets[dev] = self.gpu_manager.gpu_target_concurrent[gpu_id]
        return targets

    @property
    def gpu_target_concurrency(self):
        if self.use_gpu:
            return self.gpu_manager.get_total_capacity()
        else:
            return 0
    
    @property
    def cores_per_job(self):
        return self.cpu_manager.cores_per_job

    def adjust_target_concurrency(self):
        self.cpu_manager.adjust_concurrency()
        if self.use_gpu:
            self.gpu_manager.adjust_concurrency()
        return self.target_concurrency_split_by_resource 