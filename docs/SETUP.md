# Setup Instructions

## Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended for faster training)
- 8GB+ RAM (recommended)

## Installation

1. **Clone the repository** (if not already done):
   ```bash
   git clone <repository-url>
   cd CSM117-Final-Project
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify installation**:
   ```bash
   python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
   python -c "import torchvision; print(f'Torchvision version: {torchvision.__version__}')"
   ```

## Quick Start

### 1. Run a Simple Experiment

```bash
# Run the simple example
python examples/simple_experiment.py
```

### 2. Run Experiments from Configuration

```bash
# Run baseline experiment
python experiments/run_experiments.py --experiment baseline

# Run non-IID experiment
python experiments/run_experiments.py --experiment non_iid

# Run comparison of unlearning strategies
python experiments/run_experiments.py --experiment comparison
```

### 3. Run Custom Experiment

```bash
# Create your own config file
cp experiments/configs/baseline.yaml experiments/configs/my_experiment.yaml
# Edit my_experiment.yaml with your settings

# Run with custom config
python experiments/run_experiments.py --config experiments/configs/my_experiment.yaml
```

## Project Structure

```
CSM117-Final-Project/
├── src/                    # Source code
│   ├── fl/                # Federated learning components
│   │   ├── base.py        # Abstract base classes
│   │   └── implementations.py  # Concrete implementations
│   ├── data/              # Data handling
│   │   └── dataset_manager.py
│   ├── unlearning/        # Unlearning strategies
│   │   └── strategies.py
│   ├── evaluation/        # Evaluation metrics
│   │   └── metrics.py
│   └── utils/             # Utilities
│       └── experiment_runner.py
├── experiments/           # Experiment scripts and configs
│   ├── run_experiments.py
│   └── configs/
├── examples/              # Example scripts
│   └── simple_experiment.py
├── notebooks/             # Jupyter notebooks for analysis
│   └── analysis.ipynb
├── results/              # Experiment results (created automatically)
├── docs/                 # Documentation
├── tests/                # Unit tests
├── requirements.txt      # Dependencies
└── README.md            # This file
```

## Configuration

Experiments are configured using YAML files. Key configuration options:

### Dataset Configuration
- `dataset_name`: "cifar10", "cifar100", "mnist"
- `num_clients`: Number of federated learning clients
- `data_distribution`: "iid", "non_iid", "pathological"
- `non_iid_alpha`: Dirichlet parameter for non-IID distribution

### Model Configuration
- `model_name`: "simple_cnn", "resnet18"
- `num_classes`: Number of output classes

### Federated Learning Configuration
- `num_rounds`: Number of FL training rounds
- `local_epochs`: Local training epochs per round
- `batch_size`: Batch size for training
- `learning_rate`: Learning rate
- `aggregation_method`: "fedavg", "fedprox"

### Unlearning Configuration
- `unlearning_strategy`: "gradient_negation", "sisa", "knowledge_distillation", "fisher_information"
- `forget_ratio`: Fraction of data to forget
- `unlearning_epochs`: Number of unlearning epochs

## Available Unlearning Strategies

1. **Gradient Negation**: Performs gradient ascent on forget data
2. **SISA**: Sliced Inverse Regression for Selective Amnesia
3. **Knowledge Distillation**: Uses teacher-student framework
4. **Fisher Information**: Uses Fisher Information Matrix

## Evaluation Metrics

The framework evaluates:
- **Forget Effectiveness**: How well the model forgets the target data
- **Retention Preservation**: How well the model retains non-target data
- **Generalization Preservation**: How well the model generalizes to test data
- **Computational Cost**: Time and memory usage

## Analysis and Visualization

Use the Jupyter notebook for detailed analysis:

```bash
jupyter notebook notebooks/analysis.ipynb
```

The notebook provides:
- Training curve visualization
- Unlearning effectiveness analysis
- Comparison between experiments
- Summary report generation

## Troubleshooting

### Common Issues

1. **CUDA out of memory**:
   - Reduce `batch_size` in configuration
   - Use CPU by setting `device: "cpu"` in config

2. **Dataset download issues**:
   - Check internet connection
   - Ensure sufficient disk space
   - Try running with `data_dir: "/tmp/data"`

3. **Import errors**:
   - Ensure you're in the project root directory
   - Check that all dependencies are installed
   - Verify Python path includes `src/`

### Performance Tips

1. **Use GPU**: Set `device: "cuda"` in configuration
2. **Adjust batch size**: Larger batches for better GPU utilization
3. **Reduce rounds**: Use fewer rounds for quick testing
4. **Use smaller models**: Start with `simple_cnn` before `resnet18`

## Contributing

1. Follow the existing code structure
2. Add tests for new functionality
3. Update documentation
4. Use type hints and docstrings

## Research Timeline

- **Week 1**: Background study and problem framing
- **Week 2**: Baseline setup and experimental plan
- **Week 3**: Implementation of proposed extension
- **Week 4**: Validation and benchmarking
- **Week 5**: Final analysis and presentation

## Support

For questions or issues:
1. Check this documentation
2. Review the example scripts
3. Examine the Jupyter notebook
4. Check the experiment logs in `results/`