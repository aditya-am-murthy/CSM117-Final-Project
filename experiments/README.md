# Unlearning Experiments

This directory contains the implementation and configuration for two main unlearning experiments:

1. **Mini-batch Forgetting** with dynamic pruning and gradient replay buffers
2. **Multi-objective Learning Frameworks** using Pareto optimization

## Experiment 1: Mini-batch Forgetting

### Dynamic Pruning
- **Location**: `src/unlearning/experiments/mini_batch_forgetting/dynamic_pruning.py`
- **Description**: Adaptively isolates forget regions by dynamically pruning parameters that are most influenced by forget data
- **Key Parameters**:
  - `pruning_ratio`: Ratio of parameters to prune (default: 0.1)
  - `importance_threshold`: Threshold for parameter importance (default: 0.5)
  - `fine_tune_epochs`: Number of fine-tuning epochs on retain data (default: 5)

### Gradient Replay Buffer
- **Location**: `src/unlearning/experiments/mini_batch_forgetting/gradient_replay.py`
- **Description**: Uses a gradient replay buffer to adaptively isolate forget regions through gradient interference
- **Key Parameters**:
  - `buffer_size`: Size of gradient replay buffer (default: 100)
  - `replay_weight`: Weight for replay gradients (default: 0.5)
  - `adaptive_threshold`: Threshold for identifying forget regions (default: 0.1)

## Experiment 2: Pareto Optimization

- **Location**: `src/unlearning/experiments/pareto_optimization/pareto_unlearning.py`
- **Description**: Jointly optimizes forgetting accuracy and retention fidelity using Pareto optimization
- **Key Parameters**:
  - `forget_weight`: Weight for forgetting objective (default: 0.5)
  - `retention_weight`: Weight for retention objective (default: 0.5)
  - `adaptive_weights`: Whether to adaptively adjust weights (default: True)
  - `pareto_steps`: Number of Pareto optimization steps (default: 20)

## Running Experiments

### Prerequisites

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up wandb (optional but recommended):
```bash
wandb login
```

### Running an Experiment

Use the `run_experiments.py` script with a configuration file:

```bash
python experiments/run_experiments.py --config experiments/configs/mini_batch_forgetting_dynamic_pruning.json --verbose
```

- Pass `--verbose` (or `"verbose": true` in the config) to see per-batch/epoch logs, dataset split sizes, and evaluation summaries.

Available configuration files:
- `experiments/configs/mini_batch_forgetting_dynamic_pruning.json`
- `experiments/configs/mini_batch_forgetting_gradient_replay.json`
- `experiments/configs/pareto_optimization.json`
- `experiments/configs/report_suite.json` (runs the full set of report experiments)

### Report Experiment Suite

To reproduce the ablations described in the project report (forget ratios of 5–30%, pruning ratios of 10–50%, replay buffers of 500–5,000, and Pareto weight sweeps), run:

```bash
python experiments/run_experiments.py --config experiments/configs/report_suite.json --no-wandb
```

The new configuration format defines an `experiment_suite` with parameter grids. The runner automatically creates one experiment per grid combination, enforces deterministic stratified splits (seed 42), and writes an aggregated summary to `results/report_core_suite/suite_summary.json`.

### Configuration Format

Each configuration file is a JSON file with the following structure:

```json
{
  "experiment_name": "experiment_name",
  "description": "Description of the experiment",
  "dataset_name": "cifar10",
  "model_name": "nateraw/vit-base-patch16-224-cifar10",
  "use_vit": true,
  "num_classes": 10,
  "batch_size": 32,
  "learning_rate": 0.0001,
  "device": "cuda",
  "unlearning_strategy": "dynamic_pruning",
  "forget_ratio": 0.1,
  "test_ratio": 0.2,
  "wandb_project": "unlearning-experiments",
  "results_dir": "./results"
}
```

### Results

Results are saved in two locations:

1. **JSON Log File**: `results/{experiment_id}/experiment_log.json`
   - Contains all experiment metrics, configuration, and history
   - Includes model accuracy on forget, retain, and test sets
   - Includes unlearning effectiveness metrics

2. **Weights & Biases**: Logged to the specified wandb project
   - Real-time metrics during training
   - Visualizations and comparisons

### Metrics Logged

- **Original Model Metrics**:
  - `original_forget_accuracy`: Accuracy on forget set before unlearning
  - `original_retain_accuracy`: Accuracy on retain set before unlearning
  - `original_test_accuracy`: Accuracy on test set before unlearning

- **Unlearned Model Metrics**:
  - `unlearned_forget_accuracy`: Accuracy on forget set after unlearning
  - `unlearned_retain_accuracy`: Accuracy on retain set after unlearning
  - `unlearned_test_accuracy`: Accuracy on test set after unlearning

- **Unlearning Effectiveness**:
  - `forget_accuracy_drop`: Drop in accuracy on forget set (higher is better)
  - `retain_accuracy_drop`: Drop in accuracy on retain set (lower is better)
  - `test_accuracy_drop`: Drop in accuracy on test set (lower is better)
  - `forget_effectiveness`: Measure of how well forgetting worked
  - `retention_preservation`: Measure of how well retention was preserved
  - `generalization_preservation`: Measure of how well generalization was preserved

## Experiment Design

The experiments follow a proper unlearning evaluation protocol:

1. **Base Model**: Load a ViT model pretrained on ImageNet (NOT fine-tuned on CIFAR-10)
   - This ensures the model hasn't seen CIFAR-10 before our controlled experiment
2. **Data Split**: Split training data into forget and retain sets BEFORE training
3. **Original Model**: Fine-tune base model on ALL data (forget + retain) - this is what we want to unlearn from
4. **Gold Standard Model**: Fine-tune base model on ONLY retain data - this is the ideal result (what we want to achieve)
5. **Unlearned Model**: Apply unlearning strategy to the original model
6. **Evaluation**: Compare unlearned model to both original and gold standard

This design ensures we can properly evaluate unlearning effectiveness by:
- Starting from a model that hasn't seen CIFAR-10 (ImageNet-pretrained only)
- Training on controlled data splits where we know exactly what each model saw
- Comparing against a ground truth (gold standard model)

## Model

The experiments use the Vision Transformer (ViT) model:
- **Base Model**: `google/vit-base-patch16-224` (pretrained on ImageNet-21k, **NOT** fine-tuned on CIFAR-10)
- **Why this matters**: Using a model that hasn't seen CIFAR-10 ensures we can properly evaluate unlearning
- **Dataset**: CIFAR-10
- **Input Size**: 224x224 (resized from 32x32)
- **Training**: Models are fine-tuned on the specific data splits before unlearning
- **Classifier**: The final classification head is replaced to match CIFAR-10's 10 classes

**Important**: We use `google/vit-base-patch16-224` (ImageNet-pretrained) rather than `nateraw/vit-base-patch16-224-cifar10` (CIFAR-10 fine-tuned) because:
- The CIFAR-10 fine-tuned model already "knows" CIFAR-10, making unlearning evaluation invalid
- Starting from ImageNet-pretrained weights gives us a clean baseline that hasn't seen CIFAR-10
- We then fine-tune on our controlled data splits, allowing proper unlearning evaluation

## Customization

To create a custom experiment:

1. Create a new configuration file in `experiments/configs/`
2. Adjust parameters as needed
3. Run with: `python experiments/run_experiments.py --config your_config.json`

To disable wandb logging:
```bash
python experiments/run_experiments.py --config your_config.json --no-wandb
```

