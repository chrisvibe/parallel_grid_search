#!/usr/bin/env python3
"""
Simple example demonstrating CPU+GPU parallel training using the generic grid search framework.
This trains a single linear layer on synthetic data with different hyperparameters.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from time import time
import logging
import pandas as pd
from copy import deepcopy
from shutil import rmtree
from pathlib import Path
import yaml
from pydantic import BaseModel

# Import the grid search framework
from train_model_parallel import generic_parallel_grid_search
from parallel_utils import JobInterface

logger = logging.getLogger(__name__)


class SimpleLinearModel(nn.Module):
    """Simple neural network for demonstration"""
    
    def __init__(self, input_size: int, hidden_size: int, output_size: int = 1):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
    
    def forward(self, x):
        return self.network(x)


class SyntheticDataset:
    """Simple synthetic dataset for regression"""
    
    def __init__(self, n_samples=1000, n_features=10, noise=0.1, device='cpu'):
        self.n_samples = n_samples
        self.n_features = n_features
        self.noise = noise
        
        # Generate data
        self.X = torch.randn(n_samples, n_features)
        # Simple linear relationship with noise
        true_weights = torch.randn(n_features, 1)
        self.y = self.X @ true_weights + noise * torch.randn(n_samples, 1)
        
        # Create data loader
        dataset = torch.utils.data.TensorDataset(self.X, self.y)
        self.dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=32, shuffle=True
        )
    
    def to(self, device):
        """Move dataset to device"""
        self.X = self.X.to(device)
        self.y = self.y.to(device)
        # Recreate dataloader with moved data
        dataset = torch.utils.data.TensorDataset(self.X, self.y)
        self.dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=32, shuffle=True
        )
        return self

class SimpleParams(BaseModel):
    """Parameter structure for the simple example"""
    out_path: Path = Path("out/simple_grid_search_results")
    learning_rate: float = 0.01
    hidden_size: int = 64
    epochs: int = 50
    batch_size: int = 32
    n_samples: int = 1000
    n_features: int = 10
    noise: float = 0.1
    seed: int = 42
    test_oom: bool = True
    test_oom_job_id: str = '5-1'

class SimpleLinearJob(JobInterface):
    """Job implementation for simple linear model training"""
    
    def __init__(self, i: int, j: int, total_configs: int, total_samples: int, 
                 shared: dict, locks: dict, params: SimpleParams):
        super().__init__(i, j, total_configs, total_samples, shared, locks)
        self.params = params
    
    def _run(self, device):
        """Run training job on specified device"""
        logger.info(f"{self.get_log_prefix()} Starting on {device}")

        if self.params.test_oom and (self.params.test_oom_job_id == str(self)):
            self.params.test_oom = False
            raise torch.cuda.OutOfMemoryError("Fake OOM for testing")
        
        # Set seed for reproducibility
        torch.manual_seed(self.params.seed)
        if device.type == 'cuda':
            torch.cuda.manual_seed(self.params.seed)
        
        start_time = time()
        
        # Create dataset
        with self.locks['dataset_lock']:
            dataset = SyntheticDataset(
                n_samples=self.params.n_samples,
                n_features=self.params.n_features,
                noise=self.params.noise,
                device=device
            ).to(device)
        
        # Create model
        model = SimpleLinearModel(
            input_size=self.params.n_features,
            hidden_size=self.params.hidden_size
        ).to(device)
        
        # Train model
        final_loss, final_accuracy = self._train_model(model, dataset)
        
        training_time = time() - start_time
        
        # Create results
        results = {
            'loss': final_loss,
            'accuracy': final_accuracy,  # For compatibility with framework
            'training_time': training_time,
            'device': str(device),
            'params': self.params,
        }
        
        logger.info(f"{self.get_log_prefix()} Completed: Loss={final_loss:.4f}, Time={training_time:.2f}s")
        
        # Update shared results (similar to BooleanReservoirJob)
        with self.locks['history']:
            self.shared['history'].append({
                'config': self.i + 1,
                'sample': self.j + 1,
                'device': str(device),
                **results,
            })
        
        return {'status': 'completed', 'stats': results}
    
    def _train_model(self, model, dataset):
        """Train the model and return metrics"""
        optimizer = optim.Adam(model.parameters(), lr=self.params.learning_rate)
        criterion = nn.MSELoss()
        
        model.train()
        final_loss = 0
        
        for epoch in range(self.params.epochs):
            epoch_loss = 0
            for batch_X, batch_y in dataset.dataloader:
                optimizer.zero_grad()
                predictions = model(batch_X)
                loss = criterion(predictions, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            final_loss = epoch_loss / len(dataset.dataloader)
            
            # Log progress occasionally
            if (epoch + 1) % 20 == 0:
                logger.debug(f"{self.get_log_prefix()} Epoch {epoch+1}/{self.params.epochs}, Loss: {final_loss:.4f}")
        
        # Calculate "accuracy" (for framework compatibility - using R²)
        model.eval()
        with torch.no_grad():
            all_predictions = []
            all_targets = []
            for batch_X, batch_y in dataset.dataloader:
                pred = model(batch_X)
                all_predictions.append(pred)
                all_targets.append(batch_y)
            
            predictions = torch.cat(all_predictions, dim=0)
            targets = torch.cat(all_targets, dim=0)
            
            # Calculate R² as "accuracy"
            ss_res = torch.sum((targets - predictions) ** 2)
            ss_tot = torch.sum((targets - torch.mean(targets)) ** 2)
            r2 = 1 - (ss_res / ss_tot)
            accuracy = max(0, r2.item())  # Ensure non-negative
        
        return final_loss, accuracy

def save_grid_search_results(df: pd.DataFrame, path: Path):
    """Save grid search results to YAML, with Path fields handled by custom representers"""
    df_copy = df.copy()
    df_copy['params'] = df_copy['params'].apply(lambda p: p.model_dump())
    records = df_copy.to_dict(orient='records')
    yaml.dump_all(records, path.open('a'), sort_keys=False, Dumper=yaml.Dumper)

def load_grid_search_results(path: Path, convert=True) -> pd.DataFrame:
    """Load grid search results from YAML and reconstruct Params"""
    records = list(yaml.safe_load_all(path.open()))
    df = pd.DataFrame(records)
    df['params'] = df['params'].apply(lambda d: SimpleParams(**d))
    if convert:
        df = df.convert_dtypes()
    return df

def simple_job_factory(param_combinations):
    """Factory function to create SimpleLinearJob instances"""
    def create_job(i, j, total_configs, total_samples, shared, locks):
        params = param_combinations[i]
        # Create a copy with unique seed for this sample
        params_copy = deepcopy(params)
        params_copy.seed = params.seed + i * 1000 + j
        
        return SimpleLinearJob(
            i=i, j=j,
            total_configs=total_configs,
            total_samples=total_samples,
            shared=shared,
            locks=locks,
            params=params_copy
        )
    return create_job


def create_parameter_combinations():
    """Create parameter grid for demonstration"""
    learning_rates = [0.001, 0.01, 0.1]
    hidden_sizes = [32, 64, 128]
    
    combinations = []
    for lr in learning_rates:
        for hidden_size in hidden_sizes:
            params = SimpleParams(
                learning_rate=lr,
                hidden_size=hidden_size,
            )
            combinations.append(params)
    return combinations


def simple_parallel_grid_search(
    output_path: str = "simple_grid_search_results",
    samples_per_config: int = 3,
    gpu_memory_per_job_gb: float = 0.5,
    cpu_memory_per_job_gb: float = 1.0,
    cpu_cores_per_job: int = 1
):
    """Run parallel grid search for simple linear model"""
    
    # Create parameter combinations
    param_combinations = create_parameter_combinations()
    logger.info(f"Created {len(param_combinations)} parameter combinations")
    
    # Create job factory
    factory = simple_job_factory(param_combinations)
    
    # Define callbacks
    def save_config(output_path):
        """Save configuration info"""
        config_info = {
            'total_configs': len(param_combinations),
            'samples_per_config': samples_per_config,
            'parameter_grid': [params.__dict__ for params in param_combinations]
        }
        
        config_file = output_path / 'config.txt'
        with open(config_file, 'w') as f:
            f.write(f"Simple Linear Model Grid Search\n")
            f.write(f"Total configurations: {config_info['total_configs']}\n")
            f.write(f"Samples per configuration: {config_info['samples_per_config']}\n")
            f.write(f"Total jobs: {config_info['total_configs'] * config_info['samples_per_config']}\n\n")
            f.write("Parameter combinations:\n")
            for i, params in enumerate(config_info['parameter_grid']):
                f.write(f"Config {i+1}: {params}\n")
    
    def process_results(history, output_path: Path, done: bool):
        """Process and save results incrementally"""
        results_file = output_path / 'results.yaml'
        df = pd.DataFrame(history)
        save_grid_search_results(df, results_file)
        
        if done and results_file.exists():
            full_df = load_grid_search_results(results_file)
            
            print("\n=== Grid Search Results Summary ===")
            print(f"Total jobs completed: {len(full_df)}")
            
            config_summary = full_df.groupby('config').agg({
                'loss': ['mean', 'std', 'min'],
                'accuracy': ['mean', 'std', 'max'],
                'training_time': ['mean']
            }).round(4)
            print("\nResults by configuration:")
            print(config_summary)
            
            if 'device' in full_df.columns:
                device_summary = full_df.groupby('device').agg({
                    'training_time': ['mean', 'count']
                }).round(2)
                print("\nDevice performance:")
                print(device_summary)
        
    def cleanup():
        pass
    
    # Run the generic parallel grid search
    logger.info("Starting parallel grid search...")
    return generic_parallel_grid_search(
        job_factory=factory,
        total_configs=len(param_combinations),
        samples_per_config=samples_per_config,
        output_path=output_path,
        gpu_memory_per_job_gb=gpu_memory_per_job_gb,
        cpu_memory_per_job_gb=cpu_memory_per_job_gb,
        cpu_cores_per_job=cpu_cores_per_job,
        save_config=save_config,
        process_results=process_results,
        history_write_thresh=3,
    )


if __name__ == "__main__":
    # Set up logging
    import sys
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(process)d - %(filename)s - %(message)s',
        stream=sys.stdout,
        force=True
    )
    
    # Check available devices
    print("=== Simple Linear Model Parallel Grid Search ===")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA devices: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  Device {i}: {torch.cuda.get_device_name(i)}")
    
    p = SimpleParams()
    resp = input(f"Are you sure you want to delete '{p.out_path}'? (y/N): ").strip().lower()
    if resp == 'y':
        if Path(p.out_path).exists():
            rmtree(p.out_path)
        print("Directory deleted.")
    else:
        print("Canceled.")
    
    # Run grid search
    simple_parallel_grid_search(
        output_path=p.out_path,
        samples_per_config=2,  # Reduced for demo
        gpu_memory_per_job_gb=0.5,
        cpu_memory_per_job_gb=1.0,
        cpu_cores_per_job=1
    )