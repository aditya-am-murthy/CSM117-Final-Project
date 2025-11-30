# API Reference

## Core Classes

### FLConfig
Configuration class for federated learning experiments.

```python
from src.fl.base import FLConfig

config = FLConfig(
    num_clients=10,
    num_rounds=100,
    local_epochs=5,
    batch_size=32,
    learning_rate=0.01,
    device="cuda",
    aggregation_method="fedavg",
    unlearning_strategy="gradient_negation"
)
```

### FLClient (Abstract)
Base class for federated learning clients.

```python
from src.fl.base import FLClient

class MyClient(FLClient):
    def train_local(self, global_model_state):
        # Implement local training
        pass
    
    def evaluate_local(self):
        # Implement local evaluation
        pass
```

### FLServer (Abstract)
Base class for federated learning servers.

```python
from src.fl.base import FLServer

class MyServer(FLServer):
    def aggregate_updates(self, client_updates, client_sizes):
        # Implement aggregation
        pass
    
    def select_clients(self, round_num):
        # Implement client selection
        pass
```

## Concrete Implementations

### BasicFLClient
Standard implementation of a federated learning client.

```python
from src.fl.implementations import BasicFLClient

client = BasicFLClient(
    client_id=0,
    model=my_model,
    config=fl_config
)
```

### FedAvgServer
Federated Averaging server implementation.

```python
from src.fl.implementations import FedAvgServer

server = FedAvgServer(
    global_model=my_model,
    config=fl_config
)
```

## Data Management

### DatasetManager
Handles dataset loading and distribution.

```python
from src.data.dataset_manager import DatasetManager

# Load dataset
manager = DatasetManager("cifar10", "./data")
train_dataset, test_dataset = manager.load_dataset()

# Create IID split
client_loaders = manager.create_iid_split(
    train_dataset, num_clients=10, batch_size=32
)

# Create non-IID split
client_loaders = manager.create_non_iid_split(
    train_dataset, num_clients=10, batch_size=32, alpha=0.5
)
```

### UnlearningDataSplitter
Splits data for unlearning experiments.

```python
from src.data.dataset_manager import UnlearningDataSplitter

forget_loader, retain_loader, test_loader = UnlearningDataSplitter.split_for_unlearning(
    dataset,
    forget_ratio=0.1,
    test_ratio=0.2,
    batch_size=64,
    seed=42,
    stratified=True
)
```

## Unlearning Strategies

### GradientNegationUnlearning
Performs gradient ascent on forget data.

```python
from src.unlearning.strategies import GradientNegationUnlearning

strategy = GradientNegationUnlearning(
    config=fl_config,
    unlearning_epochs=10
)

unlearned_model = strategy.unlearn(
    model=original_model,
    forget_data=forget_loader,
    retain_data=retain_loader
)
```

### SISAUnlearning
Sliced Inverse Regression for Selective Amnesia.

```python
from src.unlearning.strategies import SISAUnlearning

strategy = SISAUnlearning(
    config=fl_config,
    num_slices=4
)
```

### KnowledgeDistillationUnlearning
Uses teacher-student framework for unlearning.

```python
from src.unlearning.strategies import KnowledgeDistillationUnlearning

strategy = KnowledgeDistillationUnlearning(
    config=fl_config,
    temperature=3.0,
    alpha=0.7
)
```

### FisherInformationUnlearning
Uses Fisher Information Matrix for unlearning.

```python
from src.unlearning.strategies import FisherInformationUnlearning

strategy = FisherInformationUnlearning(
    config=fl_config,
    lambda_reg=0.1
)
```

## Evaluation

### UnlearningEvaluator
Comprehensive evaluation framework.

```python
from src.evaluation.metrics import UnlearningEvaluator

evaluator = UnlearningEvaluator(device="cuda")

# Evaluate model performance
metrics = evaluator.evaluate_model(model, test_loader)

# Evaluate unlearning effectiveness
results = evaluator.evaluate_unlearning_effectiveness(
    original_model=original_model,
    unlearned_model=unlearned_model,
    forget_loader=forget_loader,
    retain_loader=retain_loader,
    test_loader=test_loader
)

# Measure computational cost
result, cost_metrics = evaluator.measure_computational_cost(
    func=my_function, *args, **kwargs
)
```

## Experiment Management

### ExperimentConfig
Configuration for experiments.

```python
from src.utils.experiment_runner import ExperimentConfig

config = ExperimentConfig(
    experiment_name="my_experiment",
    dataset_name="cifar10",
    num_clients=10,
    unlearning_strategy="gradient_negation",
    use_wandb=True,
    save_results=True
)
```

### ExperimentRunner
Main experiment runner.

```python
from src.utils.experiment_runner import ExperimentRunner

runner = ExperimentRunner(config)
results = runner.run_experiment()
```

## Utility Functions

### Load Configuration
```python
from src.utils.experiment_runner import load_experiment_config

config = load_experiment_config("path/to/config.yaml")
```

### Run Experiment from Config
```python
from src.utils.experiment_runner import run_experiment_from_config

results = run_experiment_from_config("path/to/config.yaml")
```

## Example Usage

### Complete Experiment Workflow

```python
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / "src"))

from utils.experiment_runner import ExperimentRunner, ExperimentConfig
from data.dataset_manager import DatasetManager
from fl.implementations import FedAvgServer, BasicFLClient
from unlearning.strategies import GradientNegationUnlearning
from evaluation.metrics import UnlearningEvaluator

# 1. Create configuration
config = ExperimentConfig(
    experiment_name="my_experiment",
    dataset_name="cifar10",
    num_clients=10,
    unlearning_strategy="gradient_negation"
)

# 2. Run experiment
runner = ExperimentRunner(config)
results = runner.run_experiment()

# 3. Access results
fl_metrics = results['fl_metrics']
unlearning_evaluation = results['unlearning_evaluation']
print(f"Final accuracy: {fl_metrics['final_accuracy']:.4f}")
print(f"Forget effectiveness: {unlearning_evaluation['unlearning_metrics']['forget_effectiveness']:.4f}")
```

### Custom Unlearning Strategy

```python
from src.fl.base import UnlearningStrategy

class MyUnlearningStrategy(UnlearningStrategy):
    def __init__(self, config):
        super().__init__(config)
        # Initialize your strategy
    
    def unlearn(self, model, forget_data, retain_data):
        # Implement your unlearning method
        unlearned_model = model  # Your implementation
        return unlearned_model
    
    def evaluate_unlearning(self, model, forget_data, retain_data):
        # Implement evaluation
        return {
            'forget_accuracy': 0.0,
            'retain_accuracy': 0.0,
            'unlearning_effectiveness': 0.0
        }
```

## Configuration Options

### ExperimentConfig Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `experiment_name` | str | "fl_unlearning_experiment" | Name of the experiment |
| `dataset_name` | str | "cifar10" | Dataset to use |
| `num_clients` | int | 10 | Number of FL clients |
| `data_distribution` | str | "iid" | Data distribution type |
| `num_rounds` | int | 100 | Number of FL rounds |
| `local_epochs` | int | 5 | Local training epochs |
| `batch_size` | int | 32 | Batch size |
| `learning_rate` | float | 0.01 | Learning rate |
| `unlearning_strategy` | str | "gradient_negation" | Unlearning method |
| `forget_ratio` | float | 0.1 | Fraction of data to forget |
| `use_wandb` | bool | False | Enable Weights & Biases logging |

### FLConfig Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_clients` | int | 10 | Number of clients |
| `num_rounds` | int | 100 | Number of rounds |
| `local_epochs` | int | 5 | Local epochs |
| `batch_size` | int | 32 | Batch size |
| `learning_rate` | float | 0.01 | Learning rate |
| `device` | str | "cuda" | Device to use |
| `aggregation_method` | str | "fedavg" | Aggregation method |
| `unlearning_strategy` | str | "none" | Unlearning strategy |
