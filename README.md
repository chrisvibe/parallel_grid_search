# parallel_grid_search
cpu + gpu work in parallel

# Simple Linear Model Parallel Training Example
This example made by AI demonstrates how to use the parallel grid search framework to train a simple PyTorch model on both CPU and GPU simultaneously.

## What This Example Does

- **Proper Grid Search Integration**: Uses your `generic_parallel_grid_search` function
- **JobInterface Implementation**: Creates `SimpleLinearJob` that inherits from `JobInterface`
- **Simple Neural Network**: 2-layer network (linear → ReLU → linear) for regression
- **Synthetic Data**: Generates regression data with controllable noise
- **Parameter Grid**: Tests combinations of:
  - Learning rates: [0.001, 0.01, 0.1]
  - Hidden sizes: [32, 64, 128]
  - Multiple samples per configuration for statistical reliability
- **CPU + GPU Parallel**: Automatically utilizes both CPU and GPU through your framework
- **Proper Logging**: Integrates with your logging system
- **Results Processing**: Saves results and provides performance summaries

## Key Features

✅ **Generic Grid Search**: Uses your actual `generic_parallel_grid_search` framework
✅ **JobInterface Compliant**: Proper `_run()` method implementation with shared state
✅ **Resource Management**: Configurable GPU/CPU memory and core allocation
✅ **Statistical Sampling**: Multiple runs per configuration for robust results
✅ **Automatic Device Detection**: Seamlessly uses available CPU and GPU resources
✅ **Results Persistence**: Saves detailed results to CSV files
✅ **Performance Analysis**: Compares device performance and finds best hyperparameters

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
=== Simple Linear Model Parallel Grid Search ===
PyTorch version: 2.0.1
CUDA available: True
CUDA devices: 1
  Device 0: NVIDIA RTX 4090

Starting parallel grid search...
Created 9 parameter combinations

[Worker logs showing parallel execution across CPU and GPU]

=== Grid Search Results Summary ===
Total jobs completed: 18

Results by configuration:
                loss                    accuracy              training_time
               mean   std   min        mean   std   max        mean
config                                                              
1             0.0234 0.0012 0.0221     0.8456 0.0023 0.8478     1.23
2             0.0198 0.0008 0.0189     0.8612 0.0015 0.8625     1.45
...

Device performance:
           training_time      
                    mean count
device                       
cpu                 2.18    9
cuda               0.89     9

Best configuration found:
Loss: 0.018432
Parameters: {'learning_rate': 0.01, 'hidden_size': 64, 'epochs': 50, ...}
```

## How It Works

### 1. Job Implementation (`SimpleLinearJob`)
- **Inherits from `JobInterface`**: Proper integration with your framework
- **`_run(device)` method**: Core training logic executed on assigned device
- **Shared State Management**: Uses locks to safely update shared history and best parameters
- **Resource Handling**: Manages data movement and model placement on CPU/GPU

### 2. Parameter Grid Creation
The `create_parameter_combinations()` function generates all combinations of:
- Learning rates × Hidden sizes = 9 total configurations
- Each configuration gets multiple samples for statistical reliability

### 3. Generic Grid Search Integration
Uses your `generic_parallel_grid_search()` function with:
- **Job Factory**: Creates job instances with proper parameters
- **Resource Allocation**: Configurable GPU/CPU memory and core limits
- **Callbacks**: Custom config saving and results processing
- **Parallel Execution**: Automatic CPU+GPU job distribution

### 4. Results Processing
- **CSV Export**: Detailed results saved for further analysis
- **Statistical Summary**: Mean, std, min/max across samples
- **Device Comparison**: Performance metrics by CPU vs GPU
- **Best Configuration**: Automatically identifies optimal hyperparameters

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