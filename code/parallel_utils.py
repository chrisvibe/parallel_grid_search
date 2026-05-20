import dataclasses
import random
import time
import psutil
import pynvml
import torch
import fcntl
import os
import tempfile
from collections import defaultdict, deque
from os import environ, sched_getaffinity
from abc import ABC, abstractmethod
from typing import Callable, Dict, Any
import numpy as np

import logging
logger = logging.getLogger(__name__)


class FileBasedLock:
    """Advisory file lock using fcntl.flock().

    Unlike mp.Lock (POSIX semaphore), the kernel automatically releases
    this lock when the owning process dies (because the fd is closed).

    Picklable: stores only the path. Each process opens its own fd in
    acquire(), which is what spawn-mode workers need.
    """

    def __init__(self, path=None):
        if path is None:
            tmp = tempfile.gettempdir()
            path = os.path.join(tmp, 'dataset_creation.lock')
        self._path = path
        open(self._path, 'a').close()
        self._fd = None

    def acquire(self, blocking=True):
        self._fd = open(self._path, 'r')
        op = fcntl.LOCK_EX if blocking else (fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            fcntl.flock(self._fd.fileno(), op)
            return True
        except BlockingIOError:
            self._fd.close()
            self._fd = None
            return False

    def release(self):
        if self._fd is not None:
            fcntl.flock(self._fd.fileno(), fcntl.LOCK_UN)
            self._fd.close()
            self._fd = None

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, *exc):
        self.release()
        return False

    def __getstate__(self):
        return {'_path': self._path}

    def __setstate__(self, state):
        self._path = state['_path']
        self._fd = None


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


@dataclasses.dataclass
class JobGenerator:
    """Iterates over the (config, sample) grid and yields jobs via job_factory."""
    job_factory: Callable
    total_configs: int
    samples_per_config: int
    locks: dict = dataclasses.field(
        default_factory=lambda: {'dataset_lock': FileBasedLock()}
    )

    def __len__(self) -> int:
        return self.total_configs * self.samples_per_config

    def __iter__(self):
        for i in range(self.total_configs):
            for j in range(self.samples_per_config):
                yield self.job_factory(
                    i=i, j=j,
                    total_configs=self.total_configs,
                    total_samples=self.samples_per_config,
                    locks=self.locks,
                )
    
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
    """CPU resource manager with SLURM support"""

    def __init__(self, memory_per_job_gb=2.0, cores_per_job=1, max_cpu_usage=0.9):
        self.memory_per_job_gb = memory_per_job_gb
        self.cores_per_job = cores_per_job
        self.max_cpu_usage = max_cpu_usage

        self._detect_constraints()
        self._log_initialization()

        psutil.cpu_percent(interval=None, percpu=True)  # prime non-blocking counter
        self._usage_cache: dict = {}
        self._usage_cache_time: float = 0.0
        self._cache_ttl: float = 0.5

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
        not yet been measured by the OS.  Their estimated memory is subtracted from
        the reported available total before applying the threshold check, preventing
        over-spawning within a single _try_scale_up call.
        """
        usage = self.get_system_usage()

        effective_available = usage['memory_available_gb'] - pending_workers * self.memory_per_job_gb
        if effective_available < self.memory_per_job_gb * 1.1:
            logger.debug(
                f"Insufficient memory: {effective_available:.1f}GB effective "
                f"({usage['memory_available_gb']:.1f}GB free, {pending_workers} pending)"
            )
            return False

        if usage['cpu_percent'] > self.max_cpu_usage * 100:
            logger.debug(f"CPU saturated: {usage['cpu_percent']:.1f}% > {self.max_cpu_usage * 100:.0f}%")
            return False

        return True

    def is_memory_pressure_elevated(self) -> bool:
        """Soft gate: pause job feeding when available RAM falls below 3× per-job estimate.

        Uses psutil.virtual_memory().available, which reads /proc/meminfo. In containers
        without cgroup v2 memory accounting this reflects host-level free RAM and may not
        detect container-OOM pressure — but it is still far more meaningful than swap %.
        """
        return self.get_system_usage()['memory_available_gb'] < self.memory_per_job_gb * 3

    def is_memory_pressure_critical(self) -> bool:
        """Hard gate: trigger graceful shutdown when available RAM falls below 1.5× per-job estimate.

        Fires above the spawn gate (1.1×) so the process can save partial results before
        the OOM killer strikes. Same cgroup caveat as is_memory_pressure_elevated.
        """
        return self.get_system_usage()['memory_available_gb'] < self.memory_per_job_gb * 1.5
   
    def handle_oom(self):
        """Handle out-of-memory error by tightening the memory gate"""
        self.memory_per_job_gb *= 1.2
        logger.warning(f"OOM detected: increased memory estimate to {self.memory_per_job_gb:.1f}GB")
    
    def get_status(self):
        """Get current system resource status string."""
        usage = self.get_system_usage()
        env_type = "SLURM" if self.is_slurm else "SYSTEM"

        status = (f"{env_type}: "
                  f"{usage['memory_available_gb']:.1f}GB free | "
                  f"{usage['memory_percent']:.0f}% mem | "
                  f"{usage['cpu_percent']:.0f}% cpu")

        if usage.get('cpu_per_core'):
            status += f" | cores: {grouped_bar_graph(usage['cpu_per_core'])}"

        return status
    

class GPUJobResourceManager:
    def __init__(self, memory_per_job_gb=2.0, buffer_gb=0.1, initial_concurrency=2):
        self.memory_per_job_gb = memory_per_job_gb
        self.buffer_gb = buffer_gb
        self.initial_concurrency = initial_concurrency

        self.gpu_metrics = defaultdict(lambda: {
            'memory_util':  {'history': deque(maxlen=10), 'ewma': 0.0},
            'compute_util': {'history': deque(maxlen=10), 'ewma': 0.0},
            'last_sample_time': 0,
        })
        self.sample_interval = 1.0
        self.ewma_alpha = 0.3

        # Raw memory read cache — avoids a pynvml call on every dispatch decision.
        # TTL is half the EWMA sample_interval so the cache is always fresher than
        # the smoothed metrics but still prevents a pynvml call per scheduler tick.
        self._mem_cache: dict = {}             # gpu_id -> {'total', 'used', 'free', 'ts'}
        self._mem_cache_ttl: float = self.sample_interval * 0.5

        # Thresholds
        self.memory_util_threshold = 0.85
        self.compute_lo_threshold = 0.90   # unsaturate below this
        self.compute_hi_threshold = 0.95   # saturate at or above this
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
            if torch.cuda.is_available():
                self.available_gpus = list(range(torch.cuda.device_count()))
                self._physical_id = {i: i for i in self.available_gpus}
            else:
                raise RuntimeError("No GPUs available")

    def _handle(self, gpu_id):
        return pynvml.nvmlDeviceGetHandleByIndex(self._physical_id[gpu_id])

    def _init_pynvml(self):
        """Initialize NVIDIA ML library and GPU tracking"""
        try:
            pynvml.nvmlInit()
        except Exception as e:
            raise RuntimeError(f"pynvml initialization failed: {e}")
        
        failed_gpus = []
        for gpu_id in self.available_gpus:
            try:
                handle = self._handle(gpu_id)
                device_name = pynvml.nvmlDeviceGetName(handle)
                if isinstance(device_name, bytes):
                    device_name = device_name.decode('utf-8')
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                
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
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                total, used, free = mem_info.total, mem_info.used, mem_info.free
                self._mem_cache[gpu_id] = {'total': total, 'used': used, 'free': free, 'ts': current_time}

            # EWMA update: fires at sample_interval regardless of whether cache was hit.
            if current_time - metrics['last_sample_time'] >= self.sample_interval:
                mem_frac = used / total if total > 0 else 0
                self._update_ewma_metric(gpu_id, 'memory_util', mem_frac)
                try:
                    handle = self._handle(gpu_id)
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
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

        try:
            with torch.cuda.device(gpu_id):
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except Exception as e:
            logger.error(f"Failed to clear cache on GPU {gpu_id}: {e}")

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
        self._last_time         = {}  # device → elapsed seconds of last completion
        self._last_update       = {}  # device → unix timestamp of last completion
        self._concurrency_limit = {}  # GPU device → AIMD-controlled worker concurrency limit

    def record_completion(self, device: torch.device, elapsed: float):
        self._last_time[device]   = elapsed
        self._last_update[device] = time.time()
        if device.type == 'cuda':
            # AIMD: compare GPU exec time against CPU exec time (apples-to-apples).
            # total_time (queue-wait + exec) is intentionally not used here — it inflates
            # the GPU metric and causes the multiplicative arm to always fire, killing workers.
            # GPU memory/compute gates already throttle concurrency; only grow the limit.
            cpu_t   = next((self._last_time[d] for d in self._last_time if d.type == 'cpu'), None)
            current = self._concurrency_limit.get(device, 1)
            if cpu_t is None or elapsed < cpu_t:
                self._concurrency_limit[device] = current + 1       # additive increase

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

    def _get_effective_weight(self, device: torch.device, default_t: float) -> float:
        last_t = self._last_time.get(device)
        if last_t is None:
            return 1.0 / max(default_t, 1e-6)
        silence = max(time.time() - self._last_update.get(device, 0), 1e-6)
        # Penalise proportionally once silence exceeds last completion time
        penalty = min(1.0, last_t / silence)
        return (1.0 / max(last_t, 1e-6)) * penalty




class ComputeJobResourceManager:
    def __init__(self, cpu_memory_per_job_gb=2.0, cpu_cores_per_job=1, gpu_memory_per_job_gb=2.0,
                 exploration_rate: float = 0.1, cpu_only: bool = False):
        self.cpu_manager = CPUJobResourceManager(
            memory_per_job_gb=cpu_memory_per_job_gb,
            cores_per_job=cpu_cores_per_job,
        )

        self.use_gpu = torch.cuda.is_available() and not cpu_only
        self.gpu_manager = None
        if self.use_gpu:
            self.gpu_manager = GPUJobResourceManager(memory_per_job_gb=gpu_memory_per_job_gb)

        # Pre-build device objects once — reused in hot path instead of allocating per call
        self._cpu_device = torch.device('cpu')
        self._gpu_devices = (
            {gpu_id: torch.device(f'cuda:{gpu_id}') for gpu_id in self.gpu_manager.available_gpus}
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

    def can_spawn_worker(self, device: torch.device, pending_workers: int = 0) -> bool:
        """Spawn gate: CPU check (all workers) + GPU check (CUDA workers only).

        pending_workers: workers spawned in this round not yet visible to the OS.
        Their estimated memory is pre-subtracted from available RAM before the
        CPU threshold is evaluated.
        """
        if not self.cpu_manager.has_available_cpu(pending_workers=pending_workers):
            return False
        if device.type == 'cuda' and self.gpu_manager is not None:
            if not self.gpu_manager._can_allocate_on_gpu(device.index):
                return False
        return True

    def handle_oom(self, device):
        if device.type == 'cuda':
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
        return self.cpu_manager.available_cpus

    @property
    def cores_per_job(self):
        return self.cpu_manager.cores_per_job