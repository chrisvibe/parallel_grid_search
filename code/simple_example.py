#!/usr/bin/env python3
"""
Simple example demonstrating CPU+GPU parallel training using the job interface.
This trains a single linear layer on synthetic data with different hyperparameters.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from dataclasses import dataclass
from typing import Dict, Any
import time
import os

# Import the job interface from your existing code
from code.parallel_utils import JobManager, Job
from code.train_model_parallel import BaseTrainingJob


@dataclass
class SimpleLinearJob(Job):
    """
    A simple job that trains a linear model on synthetic data.
    Demonstrates the job interface for CPU+GPU parallel execution.
    """
    
    def __init__(self, params: Dict[str, Any]):
        super().__init__(params)
        self.learning_rate = params.get('learning_rate', 0.01)
        self.batch_size = params.get('batch_size', 32)
        self.epochs = params.get('epochs', 100)
        self.hidden_size = params.get('hidden_size', 64)
        self.device = params.get('device', 'cpu')
        
    def create_synthetic_data(self, n_samples=1000, n_features=10):
        """Generate simple synthetic regression data"""
        X = torch.randn(n_samples, n_features)
        # Simple linear relationship with some noise
        true_weights = torch.randn(n_features, 1)
        y = X @ true_weights + 0.1 * torch.randn(n_samples, 1)
        return X, y
    
    def create_model(self, input_size: int, output_size: int = 1):
        """Create a simple linear model"""
        return nn.Sequential(
            nn.Linear(input_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, output_size)
        )
    
    def train_model(self, model, X, y, device):
        """Train the model and return final loss"""
        model = model.to(device)
        X, y = X.to(device), y.to(device)
        
        # Create data loader
        dataset = torch.utils.data.TensorDataset(X, y)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True
        )
        
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()
        
        model.train()
        final_loss = 0
        
        for epoch in range(self.epochs):
            epoch_loss = 0
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                predictions = model(batch_X)
                loss = criterion(predictions, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            final_loss = epoch_loss / len(dataloader)
            
            # Print progress every 20 epochs
            if (epoch + 1) % 20 == 0:
                print(f"Device {device}, Epoch {epoch+1}/{self.epochs}, Loss: {final_loss:.4f}")
        
        return final_loss
    
    def run(self):
        """Execute the training job"""
        print(f"Starting job on {self.device} with lr={self.learning_rate}, hidden_size={self.hidden_size}")
        
        start_time = time.time()
        
        # Generate data
        X, y = self.create_synthetic_data()
        
        # Create model
        model = self.create_model(X.shape[1])
        
        # Train model
        final_loss = self.train_model(model, X, y, self.device)
        
        training_time = time.time() - start_time
        
        result = {
            'device': self.device,
            'learning_rate': self.learning_rate,
            'hidden_size': self.hidden_size,
            'batch_size': self.batch_size,
            'final_loss': final_loss,
            'training_time': training_time,
            'epochs': self.epochs
        }
        
        print(f"Job completed on {self.device}: Loss={final_loss:.4f}, Time={training_time:.2f}s")
        return result


def create_parameter_grid():
    """Create a simple parameter grid for demonstration"""
    learning_rates = [0.001, 0.01, 0.1]
    hidden_sizes = [32, 64, 128]
    
    # Determine available devices
    devices = ['cpu']
    if torch.cuda.is_available():
        devices.append('cuda')
        print(f"CUDA available! Using devices: {devices}")
    else:
        print("CUDA not available. Using CPU only.")
    
    # Create parameter combinations
    param_combinations = []
    job_id = 0
    
    for lr in learning_rates:
        for hidden_size in hidden_sizes:
            for device in devices:
                params = {
                    'job_id': job_id,
                    'learning_rate': lr,
                    'hidden_size': hidden_size,
                    'batch_size': 32,
                    'epochs': 50,  # Reduced for demo
                    'device': device
                }
                param_combinations.append(params)
                job_id += 1
    
    return param_combinations


def run_parallel_grid_search():
    """Run the parallel grid search example"""
    print("=== Simple Linear Model Parallel Training Example ===")
    print("This example demonstrates CPU+GPU parallel training using the job interface.\n")
    
    # Create parameter grid
    param_combinations = create_parameter_grid()
    print(f"Created {len(param_combinations)} parameter combinations")
    
    # Create jobs
    jobs = [SimpleLinearJob(params) for params in param_combinations]
    
    # Run jobs in parallel
    print("\nStarting parallel execution...")
    job_manager = JobManager(max_workers=len(set(p['device'] for p in param_combinations)))
    
    start_time = time.time()
    results = job_manager.submit_jobs(jobs)
    total_time = time.time() - start_time
    
    # Analyze results
    print(f"\n=== Results Summary ===")
    print(f"Total execution time: {total_time:.2f} seconds")
    print(f"Completed {len(results)} jobs")
    
    # Find best result
    best_result = min(results, key=lambda x: x['final_loss'])
    print(f"\nBest configuration:")
    print(f"  Device: {best_result['device']}")
    print(f"  Learning Rate: {best_result['learning_rate']}")
    print(f"  Hidden Size: {best_result['hidden_size']}")
    print(f"  Final Loss: {best_result['final_loss']:.6f}")
    print(f"  Training Time: {best_result['training_time']:.2f}s")
    
    # Show device comparison
    cpu_results = [r for r in results if r['device'] == 'cpu']
    gpu_results = [r for r in results if r['device'] == 'cuda']
    
    if cpu_results and gpu_results:
        avg_cpu_time = np.mean([r['training_time'] for r in cpu_results])
        avg_gpu_time = np.mean([r['training_time'] for r in gpu_results])
        speedup = avg_cpu_time / avg_gpu_time
        print(f"\nDevice Performance:")
        print(f"  Average CPU time: {avg_cpu_time:.2f}s")
        print(f"  Average GPU time: {avg_gpu_time:.2f}s")
        print(f"  GPU speedup: {speedup:.2f}x")
    
    return results


if __name__ == "__main__":
    # Run the example
    results = run_parallel_grid_search()
    
    # Optionally save results
    import json
    with open('grid_search_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to grid_search_results.json")
