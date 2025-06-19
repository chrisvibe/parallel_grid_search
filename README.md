# parallel_grid_search
cpu + gpu work in parallel

# Simple Linear Model Parallel Training Example

This example demonstrates how to use the parallel grid search framework to train a simple PyTorch model on both CPU and GPU simultaneously.

## What This Example Does

- Creates a simple neural network with one hidden layer (linear → ReLU → linear)
- Generates synthetic regression data
- Runs a grid search over different hyperparameters:
  - Learning rates: [0.001, 0.01, 0.1]
  - Hidden sizes: [32, 64, 128]
  - Devices: ['cpu', 'cuda'] (if available)
- Trains multiple configurations in parallel across CPU and GPU
- Reports the best configuration and performance comparison

## Key Features

✅ **Job-based Interface**: Uses the existing `Job` class structure
✅ **CPU + GPU Parallel**: Automatically detects CUDA and runs jobs on both devices
✅ **Simple Model**: Just a 2-layer neural network for easy understanding
✅ **Synthetic Data**: No external data dependencies
✅ **Performance Metrics**: Tracks training time and final loss
✅ **Results Analysis**: Finds best hyperparameters and compares device performance

## Requirements

```bash
pip install torch numpy
```

**Note**: This example requires the existing job interface files (`parallel_utils.py` and `train_model_parallel.py`) to be present in the `code/` directory.

## Usage

### Quick Start
```bash
python simple_example.py
```

### Expected Output
```
=== Simple Linear Model Parallel Training Example ===
CUDA available! Using devices: ['cpu', 'cuda']
Created 18 parameter combinations

Starting parallel execution...
Starting job on cpu with lr=0.001, hidden_size=32
Starting job on cuda with lr=0.001, hidden_size=32
...
Job completed on cpu: Loss=0.0234, Time=2.15s
Job completed on cuda: Loss=0.0234, Time=0.87s

=== Results Summary ===
Total execution time: 12.45 seconds
Completed 18 jobs

Best configuration:
  Device: cuda
  Learning Rate: 0.01
  Hidden Size: 64
  Final Loss: 0.018432
  Training Time: 0.92s

Device Performance:
  Average CPU time: 2.18s
  Average GPU time: 0.89s
  GPU speedup: 2.45x
```

## How It Works

### 1. Job Definition
Each training configuration is wrapped in a `SimpleLinearJob` that:
- Inherits from the base `Job` class
- Takes hyperparameters as input
- Handles data generation, model creation, and training
- Returns performance metrics

### 2. Parameter Grid
The example creates all combinations of:
- Learning rates × Hidden sizes × Available devices

### 3. Parallel Execution
Uses the `JobManager` to:
- Submit all jobs to a worker pool
- Execute CPU and GPU jobs simultaneously
- Collect and aggregate results

### 4. Results Analysis
Compares configurations to find:
- Best hyperparameters (lowest loss)
- Device performance differences
- Overall speedup from parallelization

## Extending This Example

You can easily modify this example to:

1. **Use Your Own Data**: Replace `create_synthetic_data()` with your dataset
2. **Different Models**: Modify `create_model()` for CNNs, RNNs, etc.
3. **More Hyperparameters**: Add regularization, dropout, batch size, etc.
4. **Different Metrics**: Change from MSE loss to accuracy, F1-score, etc.
5. **Multiple GPUs**: Extend device list to `['cpu', 'cuda:0', 'cuda:1', ...]`

## File Structure Integration

The example integrates directly with your existing project structure:
```
parallel_grid_search/
├── code/
│   ├── parallel_utils.py      # JobManager, base Job class (required)
│   ├── train_model_parallel.py # BaseTrainingJob (required)
│   └── __init__.py
├── simple_example.py          # This example
└── README.md                  # This file
```

The example imports and uses your job interface classes directly.
