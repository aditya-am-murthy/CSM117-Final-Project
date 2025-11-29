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
from tqdm import tqdm
import time

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
        
        # Setup logging FIRST (before GPU setup which uses logger)
        self.verbose = self.config.get('verbose', False)
        self.log_interval = self.config.get('log_interval', 10)
        logging.basicConfig(level=logging.DEBUG if self.verbose else logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG if self.verbose else logging.INFO)
        
        # Handle multi-GPU setup
        gpu_ids = config.get('gpu_ids', None)
        if gpu_ids is not None and isinstance(gpu_ids, list) and len(gpu_ids) > 0:
            # Check if CUDA_VISIBLE_DEVICES is already set
            if 'CUDA_VISIBLE_DEVICES' in os.environ:
                self.logger.warning(f"CUDA_VISIBLE_DEVICES already set to {os.environ['CUDA_VISIBLE_DEVICES']}. "
                                  f"Requested GPUs {gpu_ids} may not match visible devices.")
                # Use the already visible devices
                visible_gpus = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
                self.gpu_ids = [int(g.strip()) for g in visible_gpus if g.strip()]
                self.data_parallel_device_ids = list(range(len(self.gpu_ids)))
                self.use_multi_gpu = len(self.gpu_ids) > 1
            else:
                # Try to set CUDA_VISIBLE_DEVICES (may not work if torch already initialized CUDA)
                # Note: For best results, set CUDA_VISIBLE_DEVICES before running the script
                try:
                    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, gpu_ids))
                    self.logger.info(f"Set CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")
                    self.logger.warning("Note: CUDA_VISIBLE_DEVICES should be set before importing torch for best results. "
                                      "If you encounter NCCL errors, try: export CUDA_VISIBLE_DEVICES=1,2,3 before running.")
                except Exception as e:
                    self.logger.warning(f"Could not set CUDA_VISIBLE_DEVICES: {e}")
                
                self.gpu_ids = gpu_ids
                # After CUDA_VISIBLE_DEVICES is set, visible devices are remapped to 0, 1, 2...
                self.data_parallel_device_ids = list(range(len(gpu_ids)))
                self.use_multi_gpu = len(gpu_ids) > 1
            
            # After setting CUDA_VISIBLE_DEVICES, PyTorch will see them as 0, 1, 2, etc.
            if not torch.cuda.is_available():
                self.logger.warning("CUDA not available, falling back to CPU")
                self.device = torch.device('cpu')
                self.use_multi_gpu = False
            else:
                self.device = torch.device('cuda:0')
                # Verify we can see the expected number of GPUs
                num_visible_gpus = torch.cuda.device_count()
                expected_gpus = len(self.gpu_ids) if self.gpu_ids else 1
                if num_visible_gpus < expected_gpus:
                    self.logger.warning(f"Only {num_visible_gpus} GPU(s) visible, expected {expected_gpus}. "
                                      f"Falling back to single GPU.")
                    self.use_multi_gpu = False
                    self.data_parallel_device_ids = None
            
            # Set NCCL environment variables to help with multi-GPU communication
            if self.use_multi_gpu:
                os.environ.setdefault('NCCL_DEBUG', 'WARN')
                os.environ.setdefault('NCCL_IB_DISABLE', '1')  # Disable InfiniBand if causing issues
                os.environ.setdefault('NCCL_P2P_DISABLE', '1')  # Disable P2P if causing issues
        else:
            # Default behavior
            self.device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
            self.gpu_ids = None
            self.data_parallel_device_ids = None
            self.use_multi_gpu = False
        
        # Create experiment logger
        experiment_id = config.get('experiment_id', f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        self.experiment_logger = ExperimentLogger(experiment_id, config.get('results_dir', './results'))
        self.experiment_logger.log_config(config)
        
        # Initialize wandb
        if self.use_wandb:
            try:
                wandb.init(
                    project=config.get('wandb_project', 'unlearning-experiments'),
                    name=experiment_id,
                    config=config,
                    reinit=True
                )
                self.logger.info(f"Wandb initialized successfully for experiment: {experiment_id}")
            except Exception as e:
                self.logger.warning(f"Failed to initialize wandb: {e}. Continuing without wandb logging.")
                self.use_wandb = False
        
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
        if 'vit' in model_name.lower() or 'google' in model_name.lower() or 'nateraw' in model_name.lower() or 'mobilevit' in model_name.lower() or 'apple' in model_name.lower():
            replace_classifier = self.config.get('replace_classifier', True)
            model = load_vit_model(model_name, num_classes, str(self.device), replace_classifier=replace_classifier)
        else:
            model = create_model_from_config(model_name, num_classes, str(self.device))
        
        # Move model to device first
        if torch.cuda.is_available():
            model = model.to(self.device)
        
        # Wrap model with DataParallel if using multiple GPUs (before compilation)
        if self.use_multi_gpu and torch.cuda.is_available():
            try:
                # After CUDA_VISIBLE_DEVICES is set, devices are remapped to 0, 1, 2...
                model = nn.DataParallel(model, device_ids=self.data_parallel_device_ids)
                self.logger.info(f"Using DataParallel on physical GPUs {self.gpu_ids} (visible as {self.data_parallel_device_ids})")
            except Exception as e:
                self.logger.warning(f"Failed to set up DataParallel: {e}. Falling back to single GPU on {self.device}")
                self.use_multi_gpu = False
        
        # Compile model for faster execution (PyTorch 2.0+) - after DataParallel
        # Note: torch.compile may not work well with DataParallel, so we skip it in multi-GPU mode
        if torch.cuda.is_available() and not self.use_multi_gpu:
            try:
                if hasattr(torch, 'compile') and self.config.get('compile_model', False):
                    self.logger.info("Compiling model with torch.compile for faster execution...")
                    model = torch.compile(model, mode='reduce-overhead')
                    self.logger.info("Model compilation complete")
            except Exception as e:
                self.logger.debug(f"Model compilation not available or failed: {e}. Continuing without compilation.")
        
        return model
    
    def _copy_model(self, model: nn.Module) -> nn.Module:
        """Copy a model, handling DataParallel wrapper correctly."""
        if isinstance(model, nn.DataParallel):
            # Unwrap, copy, and rewrap
            underlying_model = model.module
            copied_model = copy.deepcopy(underlying_model)
            if self.use_multi_gpu and torch.cuda.is_available():
                copied_model = copied_model.to(self.device)
                copied_model = nn.DataParallel(copied_model, device_ids=self.data_parallel_device_ids)
            return copied_model
        else:
            copied_model = copy.deepcopy(model)
            if self.use_multi_gpu and torch.cuda.is_available():
                copied_model = copied_model.to(self.device)
            return copied_model
    
    
    def create_unlearning_strategy(self):
        """Create unlearning strategy based on configuration."""
        strategy_type = self.config.get('unlearning_strategy', 'dynamic_pruning')
        
        fl_config = FLConfig(
            num_clients=self.config.get('num_clients', 10),
            num_rounds=self.config.get('num_rounds', 100),
            local_epochs=self.config.get('local_epochs', 5),
            batch_size=self.config.get('batch_size', 16),
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
    
    def train_model(self, model: nn.Module, train_loader: DataLoader, epochs: int = 5,
                    phase_name: str = "train") -> nn.Module:
        """Train the model on the training data."""
        model.train()
        
        # Enable optimizations for faster training
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True  # Faster convolutions
            # Use mixed precision training for 2x speedup
            use_amp = self.config.get('use_amp', True)  # Automatic Mixed Precision
            scaler = torch.cuda.amp.GradScaler() if use_amp else None
        else:
            use_amp = False
            scaler = None
        
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.get('learning_rate', 0.0001))
        criterion = nn.CrossEntropyLoss()
        total_batches = len(train_loader)
        
        self.logger.info(f"Training model for {epochs} epochs ({phase_name} phase)...")
        
        # Debug: Log model info
        if self.verbose:
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            self.logger.debug(
                f"Model Info:\n"
                f"  Total parameters: {total_params:,}\n"
                f"  Trainable parameters: {trainable_params:,}\n"
                f"  Device: {self.device}\n"
                f"  Mixed Precision (AMP): {use_amp}\n"
                f"  DataLoader workers: {train_loader.num_workers}\n"
                f"  Batch size: {train_loader.batch_size}\n"
                f"  Total batches: {total_batches}"
            )
        
        # Global step counter for wandb
        global_step = 0
        
        # Model is already loaded and tested in load_model(), so we can start training immediately
        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_total = 0
            num_batches = 0
            
            # Create progress bar for this epoch
            pbar = tqdm(
                enumerate(train_loader),
                total=total_batches,
                desc=f"{phase_name} Epoch {epoch+1}/{epochs}",
                leave=True,
                ncols=100,
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
            )
            
            for batch_idx, (data, target) in pbar:
                batch_start_time = time.time()
                
                # Time data loading from DataLoader
                data_loader_start = time.time()
                # The data is already loaded here, but we can time the iteration
                data_loader_time = time.time() - data_loader_start
                
                # Time data transfer to GPU
                data_transfer_start = time.time()
                data = data.to(self.device, non_blocking=True)
                target = target.to(self.device, non_blocking=True)
                # Synchronize to ensure transfer is complete
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                data_transfer_time = time.time() - data_transfer_start
                
                # Time optimizer zero_grad
                zero_grad_start = time.time()
                optimizer.zero_grad()
                zero_grad_time = time.time() - zero_grad_start
                
                # Time forward pass
                forward_start = time.time()
                # Use mixed precision training for faster computation
                if use_amp and scaler is not None:
                    with torch.cuda.amp.autocast():
                        output = model(data)
                        loss = criterion(output, target)
                else:
                    output = model(data)
                    loss = criterion(output, target)
                forward_time = time.time() - forward_start
                
                # Time backward pass
                backward_start = time.time()
                if use_amp and scaler is not None:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                backward_time = time.time() - backward_start
                
                # Time optimizer step
                step_start = time.time()
                if use_amp and scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                step_time = time.time() - step_start
                
                total_batch_time = time.time() - batch_start_time
                
                # Debug timing information (log first batch and every 10th batch)
                if (batch_idx == 0 or batch_idx % 10 == 0) and self.verbose:
                    self.logger.debug(
                        f"[{phase_name}] Batch {batch_idx} Timing Breakdown:\n"
                        f"  Data Load: {data_load_time*1000:.2f}ms\n"
                        f"  Zero Grad: {zero_grad_time*1000:.2f}ms\n"
                        f"  Forward: {forward_time*1000:.2f}ms ({forward_time/total_batch_time*100:.1f}%)\n"
                        f"  Backward: {backward_time*1000:.2f}ms ({backward_time/total_batch_time*100:.1f}%)\n"
                        f"  Optimizer Step: {step_time*1000:.2f}ms ({step_time/total_batch_time*100:.1f}%)\n"
                        f"  Total Batch: {total_batch_time*1000:.2f}ms\n"
                        f"  Batch Size: {data.size(0)}, Loss: {loss.item():.4f}"
                    )
                
                # Log timing to wandb for first batch and periodically
                if self.use_wandb and (batch_idx == 0 or batch_idx % 50 == 0):
                    wandb.log({
                        f'{phase_name}/timing/data_load_ms': data_load_time * 1000,
                        f'{phase_name}/timing/zero_grad_ms': zero_grad_time * 1000,
                        f'{phase_name}/timing/forward_ms': forward_time * 1000,
                        f'{phase_name}/timing/backward_ms': backward_time * 1000,
                        f'{phase_name}/timing/optimizer_step_ms': step_time * 1000,
                        f'{phase_name}/timing/total_batch_ms': total_batch_time * 1000,
                        f'{phase_name}/timing/batch_idx': batch_idx,
                        f'{phase_name}/timing/epoch': epoch + 1
                    })
                
                # Calculate batch accuracy
                predictions = output.argmax(dim=1)
                batch_correct = (predictions == target).sum().item()
                batch_total = target.size(0)
                batch_accuracy = batch_correct / batch_total
                
                epoch_loss += loss.item()
                epoch_correct += batch_correct
                epoch_total += batch_total
                num_batches += 1
                global_step += 1
                
                # Update progress bar (update every batch for smooth progress)
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{batch_accuracy:.4f}',
                    'avg_loss': f'{epoch_loss/num_batches:.4f}',
                    'avg_acc': f'{epoch_correct/epoch_total:.4f}'
                }, refresh=True)
                
                # Log to wandb at batch level (but only every N batches to avoid I/O bottleneck)
                wandb_log_interval = self.config.get('wandb_log_interval', 50)  # Log every 50 batches by default
                if self.use_wandb and (batch_idx % wandb_log_interval == 0 or batch_idx == total_batches - 1):
                    wandb.log({
                        f'{phase_name}/batch_loss': loss.item(),
                        f'{phase_name}/batch_accuracy': batch_accuracy,
                        f'{phase_name}/learning_rate': optimizer.param_groups[0]['lr'],
                        f'{phase_name}/epoch': epoch + 1,
                        f'{phase_name}/global_step': global_step
                    })
                
                # Verbose logging
                if self.verbose and (self.log_interval > 0):
                    if ((batch_idx + 1) % self.log_interval == 0) or (batch_idx + 1 == total_batches):
                        self.logger.debug(
                            f"[{phase_name}] Epoch {epoch+1}/{epochs} "
                            f"Batch {batch_idx+1}/{total_batches} "
                            f"Loss: {loss.item():.4f} Acc: {batch_accuracy:.4f}"
                        )
            
            # Close progress bar for this epoch
            pbar.close()
            
            # Calculate epoch metrics
            avg_loss = epoch_loss / max(num_batches, 1)
            avg_accuracy = epoch_correct / max(epoch_total, 1)
            
            self.logger.info(
                f"[{phase_name}] Epoch {epoch+1}/{epochs} - "
                f"Loss: {avg_loss:.4f}, Accuracy: {avg_accuracy:.4f}"
            )
            
            # Log epoch-level metrics to wandb
            if self.use_wandb:
                wandb.log({
                    f'{phase_name}/epoch_loss': avg_loss,
                    f'{phase_name}/epoch_accuracy': avg_accuracy,
                    f'{phase_name}/epoch': epoch + 1
                })
        
        return model
    
    def evaluate_model_with_progress(self, model: nn.Module, test_loader: DataLoader, 
                                     dataset_name: str = "test") -> Dict[str, float]:
        """Evaluate model with progress bar and wandb logging."""
        model.eval()
        all_predictions = []
        all_targets = []
        total_loss = 0.0
        criterion = nn.CrossEntropyLoss()
        num_batches = len(test_loader)
        
        pbar = tqdm(
            test_loader,
            desc=f"Evaluating on {dataset_name}",
            leave=True,
            ncols=100,
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
        )
        
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(pbar):
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                loss = criterion(output, target)
                
                total_loss += loss.item()
                predictions = output.argmax(dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
                
                # Update progress bar
                batch_accuracy = (predictions == target).sum().item() / target.size(0)
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{batch_accuracy:.4f}'
                })
        
        pbar.close()
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support
        accuracy = accuracy_score(all_targets, all_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_targets, all_predictions, average='weighted'
        )
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'loss': total_loss / num_batches,
            'num_samples': len(all_targets)
        }
        
        # Log to wandb
        if self.use_wandb:
            wandb.log({
                f'evaluation/{dataset_name}_accuracy': accuracy,
                f'evaluation/{dataset_name}_precision': precision,
                f'evaluation/{dataset_name}_recall': recall,
                f'evaluation/{dataset_name}_f1_score': f1,
                f'evaluation/{dataset_name}_loss': metrics['loss']
            })
        
        return metrics
    
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
            batch_size=self.config.get('batch_size', 16),
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
        num_workers = self.config.get('num_workers', 4)  # Parallel data loading
        pin_memory = torch.cuda.is_available()  # Faster GPU transfer
        combined_loader = DataLoader(
            combined_dataset,
            batch_size=self.config.get('batch_size', 16),
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=num_workers > 0
        )
        
        # Create test loader
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.get('batch_size', 16),
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=num_workers > 0
        )
        
        # Train model on ALL data (forget + retain)
        self.logger.info("Training model on all data (forget + retain)...")
        training_epochs = self.config.get('training_epochs', 5)
        original_model = self.train_model(
            self._copy_model(base_model),
            combined_loader,
            epochs=training_epochs,
            phase_name="original"
        )
        
        # Also train a "gold standard" model WITHOUT forget data (for comparison)
        self.logger.info("Training gold standard model (without forget data)...")
        gold_standard_model = self.train_model(
            self._copy_model(base_model),
            retain_loader,
            epochs=training_epochs,
            phase_name="gold_standard"
        )
        
        # Evaluate original model (trained on all data)
        self.logger.info("Evaluating original model (trained on all data)...")
        original_metrics = {
            'forget': self.evaluate_model_with_progress(original_model, forget_loader, 'original_forget'),
            'retain': self.evaluate_model_with_progress(original_model, retain_loader, 'original_retain'),
            'test': self.evaluate_model_with_progress(original_model, test_loader, 'original_test')
        }
        self._log_verbose(f"Original metrics: {json.dumps(original_metrics, indent=2)}")
        
        # Evaluate gold standard model (trained without forget data)
        self.logger.info("Evaluating gold standard model (trained without forget data)...")
        gold_standard_metrics = {
            'forget': self.evaluate_model_with_progress(gold_standard_model, forget_loader, 'gold_standard_forget'),
            'retain': self.evaluate_model_with_progress(gold_standard_model, retain_loader, 'gold_standard_retain'),
            'test': self.evaluate_model_with_progress(gold_standard_model, test_loader, 'gold_standard_test')
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
        
        # Perform unlearning
        self.logger.info("Performing unlearning...")
        unlearned_model = unlearning_strategy.unlearn(
            original_model, forget_loader, retain_loader
        )
        
        # Evaluate unlearned model
        self.logger.info("Evaluating unlearned model...")
        unlearned_metrics = {
            'forget': self.evaluate_model_with_progress(unlearned_model, forget_loader, 'unlearned_forget'),
            'retain': self.evaluate_model_with_progress(unlearned_model, retain_loader, 'unlearned_retain'),
            'test': self.evaluate_model_with_progress(unlearned_model, test_loader, 'unlearned_test')
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
            # Log final unlearned metrics
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
            
            # Create and log summary table
            summary_table = wandb.Table(columns=[
                "Model", "Forget Accuracy", "Retain Accuracy", "Test Accuracy"
            ])
            summary_table.add_data(
                "Original",
                f"{original_metrics['forget']['accuracy']:.4f}",
                f"{original_metrics['retain']['accuracy']:.4f}",
                f"{original_metrics['test']['accuracy']:.4f}"
            )
            summary_table.add_data(
                "Gold Standard",
                f"{gold_standard_metrics['forget']['accuracy']:.4f}",
                f"{gold_standard_metrics['retain']['accuracy']:.4f}",
                f"{gold_standard_metrics['test']['accuracy']:.4f}"
            )
            summary_table.add_data(
                "Unlearned",
                f"{unlearned_metrics['forget']['accuracy']:.4f}",
                f"{unlearned_metrics['retain']['accuracy']:.4f}",
                f"{unlearned_metrics['test']['accuracy']:.4f}"
            )
            wandb.log({"summary/comparison_table": summary_table})
        
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
    parser.add_argument('--gpu-ids', type=str, default=None,
                       help='Comma-separated list of GPU IDs to use (e.g., "1,2,3")')
    parser.add_argument('--batch-size', type=int, default=None,
                       help='Override batch size from config')
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    if args.verbose:
        config['verbose'] = True
    
    # Override GPU IDs if provided via command line
    if args.gpu_ids:
        config['gpu_ids'] = [int(x.strip()) for x in args.gpu_ids.split(',')]
    
    # Override batch size if provided via command line
    if args.batch_size:
        config['batch_size'] = args.batch_size
    
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
