#!/usr/bin/env python3

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent / "src"))

from utils.experiment_runner import ExperimentRunner, ExperimentConfig

def main():
    """Run a simple federated learning unlearning experiment."""
    print("Starting Federated Learning Unlearning Experiment")
    print("=" * 50)
    
    config = ExperimentConfig(
        experiment_name="simple_example",
        description="Simple example of FL unlearning with CIFAR-10",
        dataset_name="cifar10",
        num_clients=5,
        data_distribution="iid",
        num_rounds=10,
        local_epochs=2,
        unlearning_strategy="gradient_negation",
        forget_ratio=0.1,
        unlearning_epochs=3,
        use_wandb=False,
        save_results=True
    )
    
    print(f"Configuration:")
    print(f"  Dataset: {config.dataset_name}")
    print(f"  Clients: {config.num_clients}")
    print(f"  Rounds: {config.num_rounds}")
    print(f"  Unlearning Strategy: {config.unlearning_strategy}")
    print(f"  Forget Ratio: {config.forget_ratio}")
    print()
    
    runner = ExperimentRunner(config)
    results = runner.run_experiment()
    
    print("\nExperiment Results:")
    print("=" * 30)
    
    fl_metrics = results['fl_metrics']
    print(f"Final FL Accuracy: {fl_metrics['final_accuracy']:.4f}")
    print(f"Final FL Loss: {fl_metrics['final_loss']:.4f}")
    print(f"Convergence Round: {fl_metrics['convergence_round']}")
    
    unlearning_metrics = results['unlearning_evaluation']['unlearning_metrics']
    print(f"\nUnlearning Effectiveness:")
    print(f"  Forget Effectiveness: {unlearning_metrics['forget_effectiveness']:.4f}")
    print(f"  Retention Preservation: {unlearning_metrics['retention_preservation']:.4f}")
    print(f"  Generalization Preservation: {unlearning_metrics['generalization_preservation']:.4f}")
    
    print(f"\nResults saved to: {runner.results_dir}")
    print("Experiment completed successfully!")


if __name__ == "__main__":
    main()
