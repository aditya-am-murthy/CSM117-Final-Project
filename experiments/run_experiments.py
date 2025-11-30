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
import random
import itertools
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import numpy as np
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


def _assign_default_experiment_id(config: Dict[str, Any]):
    """Fallback experiment_id assignment."""
    if 'experiment_id' not in config:
        prefix = config.get('experiment_name') or config.get('unlearning_strategy') or 'experiment'
        config['experiment_id'] = f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"



def _format_value_for_id(value: Any) -> str:
    """Utility to create filesystem-safe value strings."""
    if isinstance(value, float):
        return f"{value:.3f}".rstrip('0').rstrip('.').replace('.', 'p')
    return str(value).replace(' ', '_')


def _ensure_experiment_identity(config: Dict[str, Any],
                                base_name: str,
                                entry_name: str,
                                combo_details: Optional[Dict[str, Any]],
                                index: int) -> Dict[str, Any]:
    """Populate experiment_name and experiment_id if absent."""
    if 'experiment_name' not in config:
        suffix = entry_name or f"group_{index}"
        config['experiment_name'] = f"{base_name}_{suffix}"
    
    if 'experiment_id' not in config:
        parts = [config['experiment_name']]
        if combo_details:
            combo_segment = "_".join(
                f"{key}-{_format_value_for_id(value)}"
                for key, value in combo_details.items()
            )
            if combo_segment:
                parts.append(combo_segment)
        else:
            parts.append(f"run{index:03d}")
        config['experiment_id'] = "_".join(parts)
    
    return config


def _expand_experiment_suite(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Expand experiment suites into individual experiment configs."""
    suite_entries = config.get('experiment_suite')
    if not suite_entries:
        return [config]
    
    base_config = {k: v for k, v in config.items() if k != 'experiment_suite'}
    base_name = base_config.get('experiment_name', 'experiment')
    expanded_configs: List[Dict[str, Any]] = []
    
    for entry_idx, entry in enumerate(suite_entries):
        entry_base = copy.deepcopy(base_config)
        overrides = entry.get('overrides', {})
        entry_base.update(overrides)
        entry_name = entry.get('name', f"group_{entry_idx}")
        parameter_grid = entry.get('parameter_grid')
        manual_experiments = entry.get('experiments')
        
        if manual_experiments:
            for manual_idx, manual in enumerate(manual_experiments):
                final_config = copy.deepcopy(entry_base)
                final_config.update(manual)
                final_config = _ensure_experiment_identity(
                    final_config, base_name, entry_name, None, manual_idx
                )
                expanded_configs.append(final_config)
            continue
        
        if parameter_grid:
            grid_items = sorted(parameter_grid.items())
            grid_keys = [item[0] for item in grid_items]
            grid_values = [item[1] for item in grid_items]
            
            for combo_idx, combo in enumerate(itertools.product(*grid_values)):
                combo_dict = {key: value for key, value in zip(grid_keys, combo)}
                final_config = copy.deepcopy(entry_base)
                final_config.update(combo_dict)
                final_config = _ensure_experiment_identity(
                    final_config, base_name, entry_name, combo_dict, combo_idx
                )
                expanded_configs.append(final_config)
            continue
        
        # Fallback: treat entry as direct overrides
        final_config = copy.deepcopy(entry_base)
        for key, value in entry.items():
            if key in {'name', 'overrides', 'parameter_grid', 'experiments'}:
                continue
            final_config[key] = value
        
        final_config = _ensure_experiment_identity(
            final_config, base_name, entry_name, None, entry_idx
        )
        expanded_configs.append(final_config)
    
    return expanded_configs


def _save_suite_summary(original_config: Dict[str, Any],
                        suite_results: List[Dict[str, Any]]):
    """Persist aggregated suite results to disk."""
    if not suite_results:
        return
    
    suite_name = original_config.get('suite_name') or original_config.get('experiment_name', 'suite')
    results_dir = Path(original_config.get('results_dir', './results'))
    suite_dir = results_dir / f"{suite_name}_suite"
    suite_dir.mkdir(parents=True, exist_ok=True)
    
    summary_payload = []
    for entry in suite_results:
        config_snapshot = entry['config']
        result = entry['results']
        summary_payload.append({
            'experiment_id': config_snapshot.get('experiment_id'),
            'experiment_name': config_snapshot.get('experiment_name'),
            'unlearning_strategy': config_snapshot.get('unlearning_strategy'),
            'forget_ratio': config_snapshot.get('forget_ratio'),
            'pruning_ratio': config_snapshot.get('pruning_ratio'),
            'buffer_size': config_snapshot.get('buffer_size'),
            'forget_weight': config_snapshot.get('forget_weight'),
            'retention_weight': config_snapshot.get('retention_weight'),
            'seed': config_snapshot.get('seed'),
            'metrics': result.get('unlearning_effectiveness', {}),
            'original_metrics': result.get('original_metrics', {}),
            'gold_standard_metrics': result.get('gold_standard_metrics', {}),
            'unlearned_metrics': result.get('unlearned_metrics', {})
        })
    
    with open(suite_dir / "suite_summary.json", 'w') as f:
        json.dump(summary_payload, f, indent=2, default=str)


def _print_suite_table(suite_results: List[Dict[str, Any]]):
    """Pretty print suite results to stdout."""
    if not suite_results:
        return
    
    print("\n" + "=" * 70)
    print("SUITE SUMMARY")
    print("=" * 70)
    header = f"{'Experiment ID':<45} {'Forget':>8} {'Retain':>8} {'Test':>8}"
    print(header)
    print("-" * len(header))
    
    for entry in suite_results:
        config_snapshot = entry['config']
        result = entry['results']
        unlearned_metrics = result.get('unlearned_metrics', {})
        forget_acc = unlearned_metrics.get('forget', {}).get('accuracy', 0.0)
        retain_acc = unlearned_metrics.get('retain', {}).get('accuracy', 0.0)
        test_acc = unlearned_metrics.get('test', {}).get('accuracy', 0.0)
        print(f"{config_snapshot.get('experiment_id', 'unknown'):<45} "
              f"{forget_acc:>8.3f} {retain_acc:>8.3f} {test_acc:>8.3f}")
    
    print("=" * 70 + "\n")


def _print_single_summary(results: Dict[str, Any]):
    """Pretty print a single-experiment summary."""
    print("\n" + "=" * 60)
    print("EXPERIMENT SUMMARY")
    print("=" * 60)
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
    print("=" * 60 + "\n")


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
        self._set_seed(self.config.get('seed'))
        self.device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        self.verbose = self.config.get('verbose', False)
        self.log_interval = self.config.get('log_interval', 10)
        
        # Setup logging
        logging.basicConfig(level=logging.DEBUG if self.verbose else logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG if self.verbose else logging.INFO)
        
        # Log device information
        self.logger.info(f"Using device: {self.device}")
        if torch.cuda.is_available():
            self.logger.info(f"CUDA available: {torch.cuda.is_available()}")
            self.logger.info(f"CUDA device count: {torch.cuda.device_count()}")
            if self.device.type == 'cuda':
                self.logger.info(f"Current CUDA device: {torch.cuda.current_device()}")
                self.logger.info(f"CUDA device name: {torch.cuda.get_device_name(self.device)}")
                self.logger.info(f"CUDA device memory: {torch.cuda.get_device_properties(self.device).total_memory / 1024**3:.2f} GB")
            else:
                self.logger.warning(f"Config requested CUDA but device is set to {self.device.type}")
        else:
            self.logger.warning("CUDA not available - training will run on CPU (very slow!)")
        
        # Create experiment logger
        experiment_id = config.get('experiment_id', f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        self.experiment_id = experiment_id  # Store for use in run_experiment
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
        self._log_verbose(f"Initialized runner with config: {json.dumps(self.config, indent=2, default=str)}")
    
    def _set_seed(self, seed: Optional[int]):
        """Ensure determinism across frameworks."""
        if seed is None:
            return
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    def _log_verbose(self, message: str):
        """Helper to emit debug logs only when verbose is enabled."""
        if self.verbose:
            self.logger.debug(message)
    
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
                fine_tune_epochs=self.config.get('fine_tune_epochs', 5),
                forget_loss_weight=self.config.get('forget_loss_weight', 1.0),
                prune_classifier_only=self.config.get('prune_classifier_only', False)
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
    
    def train_model(self, model: nn.Module, train_loader: DataLoader, epochs: int = 5,
                    phase_name: str = "train") -> nn.Module:
        """Train the model on the training data."""
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.get('learning_rate', 0.0001))
        criterion = nn.CrossEntropyLoss()
        total_batches = len(train_loader)
        
        self.logger.info(f"Training model for {epochs} epochs...")
        for epoch in range(epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                # Verify data is on correct device
                data = data.to(self.device)
                target = target.to(self.device)
                
                # Debug: Log device info for first batch
                if batch_idx == 0 and epoch == 0:
                    self.logger.debug(f"First batch - data device: {data.device}, target device: {target.device}, model device: {next(model.parameters()).device}")
                
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
                
                if self.verbose and (self.log_interval > 0):
                    if ((batch_idx + 1) % self.log_interval == 0) or (batch_idx + 1 == total_batches):
                        self.logger.debug(
                            f"[{phase_name}] Epoch {epoch+1}/{epochs} "
                            f"Batch {batch_idx+1}/{total_batches} Loss: {loss.item():.4f}"
                        )
            
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
        self._log_verbose(f"Loaded datasets -> train: {len(train_dataset)} samples, test: {len(test_dataset)} samples")
        
        # Split training data into forget and retain BEFORE training
        forget_ratio = self.config.get('forget_ratio', 0.1)
        test_ratio = self.config.get('test_ratio', 0.2)
        
        split_seed = self.config.get('split_seed', self.config.get('seed'))
        forget_loader, retain_loader, _ = UnlearningDataSplitter.split_for_unlearning(
            train_dataset,
            forget_ratio,
            test_ratio,
            batch_size=self.config.get('batch_size', 32),
            seed=split_seed,
            stratified=self.config.get('stratified_split', True)
        )
        self._log_verbose(
            f"Split sizes -> forget: {len(forget_loader.dataset)}, retain: {len(retain_loader.dataset)}, "
            f"test: {len(test_dataset)}, seed: {split_seed}, stratified: {self.config.get('stratified_split', True)}"
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
        
        # Check for saved models
        models_dir = Path(self.config.get('models_dir', './saved_models'))
        models_dir.mkdir(parents=True, exist_ok=True)
        
        # Use the experiment_id from __init__ (or get from config if not set)
        experiment_id = getattr(self, 'experiment_id', self.config.get('experiment_id', 'default'))
        experiment_name = self.config.get('experiment_name', 'experiment')
        
        # First try exact experiment_id match
        original_model_path = models_dir / f"{experiment_id}_original_model.pt"
        gold_standard_model_path = models_dir / f"{experiment_id}_gold_standard_model.pt"
        
        # If not found, search for models with same experiment_name (without timestamp)
        # This allows reusing models from previous runs with the same experiment config
        if not original_model_path.exists() or not gold_standard_model_path.exists():
            self.logger.info(f"Models not found with exact experiment_id: {experiment_id}")
            self.logger.info(f"Searching for models with experiment_name: {experiment_name}")
            
            # Find all models matching the experiment_name pattern
            original_pattern = f"{experiment_name}_*_original_model.pt"
            gold_pattern = f"{experiment_name}_*_gold_standard_model.pt"
            
            original_matches = sorted(models_dir.glob(original_pattern), key=lambda p: p.stat().st_mtime, reverse=True)
            gold_matches = sorted(models_dir.glob(gold_pattern), key=lambda p: p.stat().st_mtime, reverse=True)
            
            if original_matches and gold_matches:
                # Use the most recent matching models
                original_model_path = original_matches[0]
                gold_standard_model_path = gold_matches[0]
                self.logger.info(f"Found saved models from previous run:")
                self.logger.info(f"  Original model: {original_model_path.name}")
                self.logger.info(f"  Gold standard model: {gold_standard_model_path.name}")
            elif original_matches:
                self.logger.warning(f"Found original model but not gold standard. Will retrain gold standard.")
                original_model_path = original_matches[0]
            elif gold_matches:
                self.logger.warning(f"Found gold standard model but not original. Will retrain original.")
                gold_standard_model_path = gold_matches[0]
        
        self.logger.info(f"Using model paths:")
        self.logger.info(f"  Original model: {original_model_path}")
        self.logger.info(f"  Gold standard model: {gold_standard_model_path}")
        
        # Train or load original model
        # Note: training_epochs only applies to original and gold standard training, NOT unlearning
        # Unlearning uses its own epoch settings (fine_tune_epochs, unlearning_epochs)
        training_epochs = self.config.get('training_epochs', 2)
        if original_model_path.exists():
            self.logger.info(f"Loading saved original model from {original_model_path}")
            original_model = copy.deepcopy(base_model)
            original_model.load_state_dict(torch.load(original_model_path, map_location=self.device))
            original_model.to(self.device)
            self.logger.info("Original model loaded successfully")
        else:
            self.logger.info("Training model on all data (forget + retain)...")
            original_model = self.train_model(
                copy.deepcopy(base_model),
                combined_loader,
                epochs=training_epochs,
                phase_name="original"
            )
            # Save original model
            torch.save(original_model.state_dict(), original_model_path)
            self.logger.info(f"Saved original model to {original_model_path}")
        
        # Train or load gold standard model
        if gold_standard_model_path.exists():
            self.logger.info(f"Loading saved gold standard model from {gold_standard_model_path}")
            gold_standard_model = copy.deepcopy(base_model)
            gold_standard_model.load_state_dict(torch.load(gold_standard_model_path, map_location=self.device))
            gold_standard_model.to(self.device)
            self.logger.info("Gold standard model loaded successfully")
        else:
            self.logger.info("Training gold standard model (without forget data)...")
            gold_standard_model = self.train_model(
                copy.deepcopy(base_model),
                retain_loader,
                epochs=training_epochs,
                phase_name="gold_standard"
            )
            # Save gold standard model
            torch.save(gold_standard_model.state_dict(), gold_standard_model_path)
            self.logger.info(f"Saved gold standard model to {gold_standard_model_path}")
        
        # Evaluate original model (trained on all data)
        self.logger.info("Evaluating original model (trained on all data)...")
        original_metrics = {
            'forget': self.evaluator.evaluate_model(original_model, forget_loader),
            'retain': self.evaluator.evaluate_model(original_model, retain_loader),
            'test': self.evaluator.evaluate_model(original_model, test_loader)
        }
        self._log_verbose(f"Original metrics: {json.dumps(original_metrics, indent=2)}")
        
        # Evaluate gold standard model (trained without forget data)
        self.logger.info("Evaluating gold standard model (trained without forget data)...")
        gold_standard_metrics = {
            'forget': self.evaluator.evaluate_model(gold_standard_model, forget_loader),
            'retain': self.evaluator.evaluate_model(gold_standard_model, retain_loader),
            'test': self.evaluator.evaluate_model(gold_standard_model, test_loader)
        }
        self._log_verbose(f"Gold standard metrics: {json.dumps(gold_standard_metrics, indent=2)}")
        
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
        
        # Perform unlearning with progress tracking
        self.logger.info("=" * 60)
        self.logger.info("Starting unlearning process...")
        self.logger.info(f"Strategy: {self.config.get('unlearning_strategy', 'unknown')}")
        self.logger.info(f"Forget data batches: {len(forget_loader)}")
        self.logger.info(f"Retain data batches: {len(retain_loader)}")
        self.logger.info("=" * 60)
        
        import time
        unlearning_start_time = time.time()
        
        try:
            unlearned_model = unlearning_strategy.unlearn(
                original_model, forget_loader, retain_loader
            )
            unlearning_time = time.time() - unlearning_start_time
            self.logger.info(f"Unlearning completed in {unlearning_time:.2f} seconds ({unlearning_time/60:.2f} minutes)")
        except Exception as e:
            self.logger.error(f"Error during unlearning: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            raise
        
        # Evaluate unlearned model
        self.logger.info("Evaluating unlearned model...")
        unlearned_metrics = {
            'forget': self.evaluator.evaluate_model(unlearned_model, forget_loader),
            'retain': self.evaluator.evaluate_model(unlearned_model, retain_loader),
            'test': self.evaluator.evaluate_model(unlearned_model, test_loader)
        }
        self._log_verbose(f"Unlearned metrics: {json.dumps(unlearned_metrics, indent=2)}")
        
        # Get unlearning strategy evaluation
        strategy_metrics = unlearning_strategy.evaluate_unlearning(
            unlearned_model, forget_loader, retain_loader
        )
        self._log_verbose(f"Strategy metrics: {json.dumps(strategy_metrics, indent=2)}")
        
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
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging for training and evaluation')
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    if args.verbose:
        config['verbose'] = True
    
    wandb_allowed = not args.no_wandb
    base_wandb_pref = config.get('use_wandb', True)
    expanded_configs = _expand_experiment_suite(config)
    
    # Ensure each config has an experiment_id
    for cfg in expanded_configs:
        _assign_default_experiment_id(cfg)
    
    suite_mode = len(expanded_configs) > 1
    suite_results: List[Dict[str, Any]] = []
    
    for cfg in expanded_configs:
        cfg_wandb_pref = cfg.get('use_wandb', base_wandb_pref)
        runner = UnlearningExperimentRunner(cfg, use_wandb=(cfg_wandb_pref and wandb_allowed))
        results = runner.run_experiment()
        _print_single_summary(results)
        suite_results.append({'config': cfg, 'results': results})
    
    if suite_mode:
        _save_suite_summary(config, suite_results)
        _print_suite_table(suite_results)


if __name__ == "__main__":
    main()
