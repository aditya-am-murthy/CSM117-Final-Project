"""
Modular training script for unlearning experiments.

This script supports:
1. Mini-batch forgetting with dynamic pruning and gradient replay buffers
2. Multi-objective learning frameworks (Pareto optimization)

All experiments are logged to JSON files and wandb.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import wandb
import logging
import copy

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.dataset_manager import DatasetManager, UnlearningDataSplitter
from src.utils.model_loader import load_vit_model, create_model_from_config
from src.evaluation.metrics import UnlearningEvaluator
from src.unlearning.experiments.mini_batch_forgetting import (
    DynamicPruningUnlearning,
    GradientReplayBufferUnlearning
)
from src.unlearning.experiments.pareto_optimization import (
    ParetoOptimizationUnlearning
)
from src.fl.base import FLConfig


class ExperimentLogger:
    """Handles JSON logging for experiments."""
    
    def __init__(self, experiment_id: str, results_dir: str = "./results"):
        self.experiment_id = experiment_id
        self.results_dir = Path(results_dir) / experiment_id
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.results_dir / "experiment_log.json"
        self.results = {
            'experiment_id': experiment_id,
            'timestamp': datetime.now().isoformat(),
            'config': {},
            'metrics': {},
            'history': []
        }
    
    def log_config(self, config: Dict[str, Any]):
        """Log experiment configuration."""
        self.results['config'] = config
    
    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None):
        """Log metrics at a specific step."""
        if step is not None:
            self.results['history'].append({
                'step': step,
                'metrics': metrics,
                'timestamp': datetime.now().isoformat()
            })
        else:
            self.results['metrics'].update(metrics)
    
    def save(self):
        """Save results to JSON file."""
        with open(self.log_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        logging.info(f"Results saved to {self.log_file}")


class UnlearningExperimentRunner:
    """Main experiment runner for unlearning experiments."""
    
    def __init__(self, config: Dict[str, Any], use_wandb: bool = True):
        self.config = config
        self.use_wandb = use_wandb
        self.device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Create experiment logger
        experiment_id = config.get('experiment_id', f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        self.experiment_logger = ExperimentLogger(experiment_id, config.get('results_dir', './results'))
        self.experiment_logger.log_config(config)
        
        # Initialize wandb
        if self.use_wandb:
            wandb.init(
                project=config.get('wandb_project', 'unlearning-experiments'),
                name=experiment_id,
                config=config
            )
        
        # Setup dataset manager
        use_vit = 'vit' in config.get('model_name', '').lower() or config.get('use_vit', False)
        self.dataset_manager = DatasetManager(
            dataset_name=config.get('dataset_name', 'cifar10'),
            data_dir=config.get('data_dir', './data'),
            use_vit=use_vit
        )
        
        # Setup evaluator
        self.evaluator = UnlearningEvaluator(device=str(self.device))
    
    def load_model(self) -> nn.Module:
        """Load the model based on configuration."""
        model_name = self.config.get('model_name', 'google/vit-base-patch16-224')
        num_classes = self.config.get('num_classes', 10)
        
        # For unlearning experiments, we want ImageNet-pretrained models, not CIFAR-10 fine-tuned
        # This ensures the model hasn't seen CIFAR-10 before our controlled training
        if 'vit' in model_name.lower() or 'google' in model_name.lower() or 'nateraw' in model_name.lower():
            replace_classifier = self.config.get('replace_classifier', True)
            model = load_vit_model(model_name, num_classes, str(self.device), replace_classifier=replace_classifier)
        else:
            model = create_model_from_config(model_name, num_classes, str(self.device))
        
        return model
    
    
    def create_unlearning_strategy(self):
        """Create unlearning strategy based on configuration."""
        strategy_type = self.config.get('unlearning_strategy', 'dynamic_pruning')
        
        fl_config = FLConfig(
            num_clients=self.config.get('num_clients', 10),
            num_rounds=self.config.get('num_rounds', 100),
            local_epochs=self.config.get('local_epochs', 5),
            batch_size=self.config.get('batch_size', 32),
            learning_rate=self.config.get('learning_rate', 0.001),
            device=str(self.device)
        )
        
        if strategy_type == 'dynamic_pruning':
            return DynamicPruningUnlearning(
                fl_config,
                pruning_ratio=self.config.get('pruning_ratio', 0.1),
                importance_threshold=self.config.get('importance_threshold', 0.5),
                fine_tune_epochs=self.config.get('fine_tune_epochs', 5)
            )
        elif strategy_type == 'gradient_replay':
            return GradientReplayBufferUnlearning(
                fl_config,
                buffer_size=self.config.get('buffer_size', 100),
                replay_weight=self.config.get('replay_weight', 0.5),
                adaptive_threshold=self.config.get('adaptive_threshold', 0.1),
                unlearning_epochs=self.config.get('unlearning_epochs', 10)
            )
        elif strategy_type == 'pareto_optimization':
            return ParetoOptimizationUnlearning(
                fl_config,
                forget_weight=self.config.get('forget_weight', 0.5),
                retention_weight=self.config.get('retention_weight', 0.5),
                pareto_steps=self.config.get('pareto_steps', 20),
                adaptive_weights=self.config.get('adaptive_weights', True),
                unlearning_epochs=self.config.get('unlearning_epochs', 10)
            )
        else:
            raise ValueError(f"Unknown unlearning strategy: {strategy_type}")
    
    def train_model(self, model: nn.Module, train_loader: DataLoader, epochs: int = 5) -> nn.Module:
        """Train the model on the training data."""
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.get('learning_rate', 0.0001))
        criterion = nn.CrossEntropyLoss()
        
        self.logger.info(f"Training model for {epochs} epochs...")
        for epoch in range(epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            for data, target in train_loader:
                data = data.to(self.device)
                target = target.to(self.device)
                
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            avg_loss = epoch_loss / max(num_batches, 1)
            self.logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
            
            if self.use_wandb:
                wandb.log({'training/epoch': epoch+1, 'training/loss': avg_loss})
        
        return model
    
    def run_experiment(self) -> Dict[str, Any]:
        """Run the complete unlearning experiment."""
        self.logger.info(f"Starting experiment: {self.config.get('experiment_id', 'unknown')}")
        
        # Load base model (pretrained, but not fine-tuned on our specific data)
        self.logger.info("Loading base model...")
        base_model = self.load_model()
        
        # Prepare data splits BEFORE training
        self.logger.info("Preparing data splits...")
        train_dataset, test_dataset = self.dataset_manager.load_dataset()
        
        # Split training data into forget and retain BEFORE training
        forget_ratio = self.config.get('forget_ratio', 0.1)
        test_ratio = self.config.get('test_ratio', 0.2)
        
        forget_loader, retain_loader, _ = UnlearningDataSplitter.split_for_unlearning(
            train_dataset, forget_ratio, test_ratio
        )
        
        # Create combined training loader (forget + retain) for initial training
        from torch.utils.data import ConcatDataset
        combined_dataset = ConcatDataset([forget_loader.dataset, retain_loader.dataset])
        combined_loader = DataLoader(
            combined_dataset,
            batch_size=self.config.get('batch_size', 32),
            shuffle=True
        )
        
        # Create test loader
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.get('batch_size', 32),
            shuffle=False
        )
        
        # Train model on ALL data (forget + retain)
        self.logger.info("Training model on all data (forget + retain)...")
        training_epochs = self.config.get('training_epochs', 5)
        original_model = self.train_model(
            copy.deepcopy(base_model),
            combined_loader,
            epochs=training_epochs
        )
        
        # Also train a "gold standard" model WITHOUT forget data (for comparison)
        self.logger.info("Training gold standard model (without forget data)...")
        gold_standard_model = self.train_model(
            copy.deepcopy(base_model),
            retain_loader,
            epochs=training_epochs
        )
        
        # Evaluate original model (trained on all data)
        self.logger.info("Evaluating original model (trained on all data)...")
        original_metrics = {
            'forget': self.evaluator.evaluate_model(original_model, forget_loader),
            'retain': self.evaluator.evaluate_model(original_model, retain_loader),
            'test': self.evaluator.evaluate_model(original_model, test_loader)
        }
        
        # Evaluate gold standard model (trained without forget data)
        self.logger.info("Evaluating gold standard model (trained without forget data)...")
        gold_standard_metrics = {
            'forget': self.evaluator.evaluate_model(gold_standard_model, forget_loader),
            'retain': self.evaluator.evaluate_model(gold_standard_model, retain_loader),
            'test': self.evaluator.evaluate_model(gold_standard_model, test_loader)
        }
        
        self.logger.info(f"Original model - Test Accuracy: {original_metrics['test']['accuracy']:.4f}")
        self.logger.info(f"Gold standard model - Test Accuracy: {gold_standard_metrics['test']['accuracy']:.4f}")
        
        self.experiment_logger.log_metrics({
            'original_forget_accuracy': original_metrics['forget']['accuracy'],
            'original_retain_accuracy': original_metrics['retain']['accuracy'],
            'original_test_accuracy': original_metrics['test']['accuracy'],
            'gold_standard_forget_accuracy': gold_standard_metrics['forget']['accuracy'],
            'gold_standard_retain_accuracy': gold_standard_metrics['retain']['accuracy'],
            'gold_standard_test_accuracy': gold_standard_metrics['test']['accuracy']
        })
        
        if self.use_wandb:
            wandb.log({
                'original/forget_accuracy': original_metrics['forget']['accuracy'],
                'original/retain_accuracy': original_metrics['retain']['accuracy'],
                'original/test_accuracy': original_metrics['test']['accuracy'],
                'gold_standard/forget_accuracy': gold_standard_metrics['forget']['accuracy'],
                'gold_standard/retain_accuracy': gold_standard_metrics['retain']['accuracy'],
                'gold_standard/test_accuracy': gold_standard_metrics['test']['accuracy']
            })
        
        # Create unlearning strategy
        self.logger.info("Creating unlearning strategy...")
        unlearning_strategy = self.create_unlearning_strategy()
        
        # Perform unlearning
        self.logger.info("Performing unlearning...")
        unlearned_model = unlearning_strategy.unlearn(
            original_model, forget_loader, retain_loader
        )
        
        # Evaluate unlearned model
        self.logger.info("Evaluating unlearned model...")
        unlearned_metrics = {
            'forget': self.evaluator.evaluate_model(unlearned_model, forget_loader),
            'retain': self.evaluator.evaluate_model(unlearned_model, retain_loader),
            'test': self.evaluator.evaluate_model(unlearned_model, test_loader)
        }
        
        # Get unlearning strategy evaluation
        strategy_metrics = unlearning_strategy.evaluate_unlearning(
            unlearned_model, forget_loader, retain_loader
        )
        
        # Compute unlearning effectiveness
        forget_accuracy_drop = original_metrics['forget']['accuracy'] - unlearned_metrics['forget']['accuracy']
        retain_accuracy_drop = original_metrics['retain']['accuracy'] - unlearned_metrics['retain']['accuracy']
        test_accuracy_drop = original_metrics['test']['accuracy'] - unlearned_metrics['test']['accuracy']
        
        # Compare to gold standard (ideal unlearning result)
        forget_gap_to_gold = unlearned_metrics['forget']['accuracy'] - gold_standard_metrics['forget']['accuracy']
        retain_gap_to_gold = unlearned_metrics['retain']['accuracy'] - gold_standard_metrics['retain']['accuracy']
        test_gap_to_gold = unlearned_metrics['test']['accuracy'] - gold_standard_metrics['test']['accuracy']
        
        results = {
            'original_metrics': original_metrics,
            'gold_standard_metrics': gold_standard_metrics,
            'unlearned_metrics': unlearned_metrics,
            'strategy_metrics': strategy_metrics,
            'unlearning_effectiveness': {
                'forget_accuracy_drop': forget_accuracy_drop,
                'retain_accuracy_drop': retain_accuracy_drop,
                'test_accuracy_drop': test_accuracy_drop,
                'forget_effectiveness': forget_accuracy_drop,
                'retention_preservation': 1.0 - retain_accuracy_drop,
                'generalization_preservation': 1.0 - test_accuracy_drop,
                # Comparison to gold standard (lower is better - means closer to ideal)
                'forget_gap_to_gold_standard': abs(forget_gap_to_gold),
                'retain_gap_to_gold_standard': abs(retain_gap_to_gold),
                'test_gap_to_gold_standard': abs(test_gap_to_gold)
            }
        }
        
        # Log results
        self.experiment_logger.log_metrics({
            'unlearned_forget_accuracy': unlearned_metrics['forget']['accuracy'],
            'unlearned_retain_accuracy': unlearned_metrics['retain']['accuracy'],
            'unlearned_test_accuracy': unlearned_metrics['test']['accuracy'],
            **results['unlearning_effectiveness']
        })
        
        if self.use_wandb:
            wandb.log({
                'unlearned/forget_accuracy': unlearned_metrics['forget']['accuracy'],
                'unlearned/retain_accuracy': unlearned_metrics['retain']['accuracy'],
                'unlearned/test_accuracy': unlearned_metrics['test']['accuracy'],
                'unlearning/forget_accuracy_drop': forget_accuracy_drop,
                'unlearning/retain_accuracy_drop': retain_accuracy_drop,
                'unlearning/test_accuracy_drop': test_accuracy_drop,
                'unlearning/forget_effectiveness': forget_accuracy_drop,
                'unlearning/retention_preservation': 1.0 - retain_accuracy_drop,
                'unlearning/generalization_preservation': 1.0 - test_accuracy_drop,
                'comparison/forget_gap_to_gold': abs(forget_gap_to_gold),
                'comparison/retain_gap_to_gold': abs(retain_gap_to_gold),
                'comparison/test_gap_to_gold': abs(test_gap_to_gold)
            })
            
            # Log strategy-specific metrics
            for key, value in strategy_metrics.items():
                if isinstance(value, (int, float)):
                    wandb.log({f'strategy/{key}': value})
        
        # Save results
        self.experiment_logger.results['final_results'] = results
        self.experiment_logger.save()
        
        self.logger.info("Experiment completed successfully!")
        self.logger.info(f"Original Model - Test Accuracy: {original_metrics['test']['accuracy']:.4f}")
        self.logger.info(f"Gold Standard - Test Accuracy: {gold_standard_metrics['test']['accuracy']:.4f}")
        self.logger.info(f"Unlearned Model - Test Accuracy: {unlearned_metrics['test']['accuracy']:.4f}")
        self.logger.info(f"Forget Effectiveness: {forget_accuracy_drop:.4f}")
        self.logger.info(f"Retention Preservation: {1.0 - retain_accuracy_drop:.4f}")
        self.logger.info(f"Gap to Gold Standard (Test): {abs(test_gap_to_gold):.4f}")
        
        if self.use_wandb:
            wandb.finish()
        
        return results


def main():
    """Main entry point for running experiments."""
    parser = argparse.ArgumentParser(description='Run unlearning experiments')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to experiment configuration JSON file')
    parser.add_argument('--no-wandb', action='store_true',
                       help='Disable wandb logging')
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    # Add experiment ID if not present
    if 'experiment_id' not in config:
        config['experiment_id'] = f"{config.get('unlearning_strategy', 'experiment')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Run experiment
    runner = UnlearningExperimentRunner(config, use_wandb=not args.no_wandb)
    results = runner.run_experiment()
    
    print("\n" + "="*60)
    print("EXPERIMENT SUMMARY")
    print("="*60)
    print(f"Original Model (trained on all data):")
    print(f"  Test Accuracy: {results['original_metrics']['test']['accuracy']:.4f}")
    print(f"\nGold Standard Model (trained WITHOUT forget data):")
    print(f"  Test Accuracy: {results['gold_standard_metrics']['test']['accuracy']:.4f}")
    print(f"\nUnlearned Model:")
    print(f"  Test Accuracy: {results['unlearned_metrics']['test']['accuracy']:.4f}")
    print(f"\nUnlearning Effectiveness:")
    print(f"  Forget Effectiveness: {results['unlearning_effectiveness']['forget_effectiveness']:.4f}")
    print(f"  Retention Preservation: {results['unlearning_effectiveness']['retention_preservation']:.4f}")
    print(f"  Generalization Preservation: {results['unlearning_effectiveness']['generalization_preservation']:.4f}")
    print(f"\nComparison to Gold Standard:")
    print(f"  Test Accuracy Gap: {results['unlearning_effectiveness']['test_gap_to_gold_standard']:.4f}")
    print("="*60)


if __name__ == "__main__":
    main()
