"""
Example experiment configurations and scripts.

This module provides example configurations and scripts for running
different types of federated learning unlearning experiments.
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.experiment_runner import ExperimentRunner, ExperimentConfig, run_experiment_from_config


def create_baseline_config() -> ExperimentConfig:
    """Create baseline experiment configuration."""
    return ExperimentConfig(
        experiment_name="baseline_experiment",
        description="Baseline federated learning with gradient negation unlearning",
        dataset_name="cifar10",
        num_clients=10,
        data_distribution="iid",
        num_rounds=50,
        local_epochs=3,
        unlearning_strategy="gradient_negation",
        forget_ratio=0.1,
        unlearning_epochs=5,
        use_wandb=False,
        save_results=True
    )


def create_non_iid_config() -> ExperimentConfig:
    """Create non-IID experiment configuration."""
    return ExperimentConfig(
        experiment_name="non_iid_experiment",
        description="Non-IID federated learning with SISA unlearning",
        dataset_name="cifar10",
        num_clients=10,
        data_distribution="non_iid",
        non_iid_alpha=0.3,
        num_rounds=50,
        local_epochs=3,
        unlearning_strategy="sisa",
        forget_ratio=0.1,
        use_wandb=False,
        save_results=True
    )


def create_comparison_config() -> ExperimentConfig:
    """Create configuration for comparing different unlearning strategies."""
    return ExperimentConfig(
        experiment_name="unlearning_comparison",
        description="Comparison of different unlearning strategies",
        dataset_name="cifar10",
        num_clients=10,
        data_distribution="iid",
        num_rounds=30,
        local_epochs=3,
        unlearning_strategy="gradient_negation",  # Will be overridden
        forget_ratio=0.1,
        use_wandb=False,
        save_results=True
    )


def run_baseline_experiment():
    """Run baseline experiment."""
    print("Running baseline experiment...")
    config = create_baseline_config()
    runner = ExperimentRunner(config)
    results = runner.run_experiment()
    print("Baseline experiment completed!")
    return results


def run_non_iid_experiment():
    """Run non-IID experiment."""
    print("Running non-IID experiment...")
    config = create_non_iid_config()
    runner = ExperimentRunner(config)
    results = runner.run_experiment()
    print("Non-IID experiment completed!")
    return results


def run_unlearning_comparison():
    """Run comparison of different unlearning strategies."""
    strategies = ["gradient_negation", "sisa", "knowledge_distillation", "fisher_information"]
    results = {}
    
    for strategy in strategies:
        print(f"Running experiment with {strategy} unlearning...")
        config = create_comparison_config()
        config.experiment_name = f"comparison_{strategy}"
        config.unlearning_strategy = strategy
        
        runner = ExperimentRunner(config)
        strategy_results = runner.run_experiment()
        results[strategy] = strategy_results
        print(f"Experiment with {strategy} completed!")
    
    return results


def run_ablation_study():
    """Run ablation study on different parameters."""
    forget_ratios = [0.05, 0.1, 0.2, 0.3]
    results = {}
    
    for forget_ratio in forget_ratios:
        print(f"Running experiment with forget_ratio={forget_ratio}...")
        config = create_baseline_config()
        config.experiment_name = f"ablation_forget_ratio_{forget_ratio}"
        config.forget_ratio = forget_ratio
        
        runner = ExperimentRunner(config)
        ratio_results = runner.run_experiment()
        results[f"forget_ratio_{forget_ratio}"] = ratio_results
        print(f"Experiment with forget_ratio={forget_ratio} completed!")
    
    return results


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Run federated learning unlearning experiments")
    parser.add_argument("--experiment", type=str, default="baseline",
                       choices=["baseline", "non_iid", "comparison", "ablation"],
                       help="Type of experiment to run")
    parser.add_argument("--config", type=str, default=None,
                       help="Path to configuration file")
    
    args = parser.parse_args()
    
    if args.config:
        # Run experiment from config file
        results = run_experiment_from_config(args.config)
    else:
        # Run predefined experiment
        if args.experiment == "baseline":
            results = run_baseline_experiment()
        elif args.experiment == "non_iid":
            results = run_non_iid_experiment()
        elif args.experiment == "comparison":
            results = run_unlearning_comparison()
        elif args.experiment == "ablation":
            results = run_ablation_study()
        else:
            raise ValueError(f"Unknown experiment type: {args.experiment}")
    
    print("All experiments completed!")
