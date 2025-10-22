"""
Experiment runner and configuration system for federated learning unlearning.

This module provides a comprehensive framework for running experiments
with different configurations and tracking results.
"""

import os
import json
import yaml
import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import logging
from datetime import datetime
import wandb
from rich.console import Console
from rich.progress import Progress, TaskID

from ..fl.base import FLConfig, FLExperiment
from ..fl.implementations import FedAvgServer, BasicFLClient
from ..data.dataset_manager import DatasetManager, UnlearningDataSplitter
from ..unlearning.strategies import (
    SISAUnlearning, GradientNegationUnlearning, 
    KnowledgeDistillationUnlearning, FisherInformationUnlearning
)
from ..evaluation.metrics import UnlearningEvaluator


@dataclass
class ExperimentConfig:
    """Configuration for experiments."""
    experiment_name: str = "fl_unlearning_experiment"
    experiment_id: str = ""
    description: str = ""
    
    dataset_name: str = "cifar10"
    data_dir: str = "./data"
    num_clients: int = 10
    data_distribution: str = "iid"
    non_iid_alpha: float = 0.5
    
    model_name: str = "resnet18"
    num_classes: int = 10
    
    num_rounds: int = 100
    local_epochs: int = 5
    batch_size: int = 32
    learning_rate: float = 0.01
    aggregation_method: str = "fedavg"
    
    unlearning_strategy: str = "gradient_negation"
    forget_ratio: float = 0.1
    test_ratio: float = 0.2
    unlearning_epochs: int = 10

    evaluation_metrics: List[str] = None
    
    use_wandb: bool = False
    wandb_project: str = "fl-unlearning"
    log_level: str = "INFO"
    save_results: bool = True
    results_dir: str = "./results"
    
    def __post_init__(self):
        if self.evaluation_metrics is None:
            self.evaluation_metrics = ["accuracy", "precision", "recall", "f1_score"]
        
        if not self.experiment_id:
            self.experiment_id = f"{self.experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


class ExperimentRunner:
    """Main experiment runner for federated learning unlearning experiments."""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.console = Console()
        self.logger = self._setup_logging()
        self.evaluator = UnlearningEvaluator()
        
        self.dataset_manager = DatasetManager(config.dataset_name, config.data_dir)
        self.model = self._create_model()
        self.fl_config = self._create_fl_config()
        
        self.results_dir = Path(config.results_dir) / config.experiment_id
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        if config.use_wandb:
            wandb.init(
                project=config.wandb_project,
                name=config.experiment_id,
                config=asdict(config)
            )
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for the experiment."""
        logger = logging.getLogger(self.config.experiment_name)
        logger.setLevel(getattr(logging, self.config.log_level))
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        log_file = self.results_dir / "experiment.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        return logger
    
    def _create_model(self) -> nn.Module:
        """Create the model based on configuration."""
        if self.config.model_name.lower() == "resnet18":
            from torchvision.models import resnet18
            model = resnet18(pretrained=False)
            model.fc = nn.Linear(model.fc.in_features, self.config.num_classes)
        elif self.config.model_name.lower() == "simple_cnn":
            model = self._create_simple_cnn()
        else:
            raise ValueError(f"Unsupported model: {self.config.model_name}")
        
        return model
    
    def _create_simple_cnn(self) -> nn.Module:
        """Create a simple CNN for CIFAR-10."""
        return nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, self.config.num_classes)
        )
    
    def _create_fl_config(self) -> FLConfig:
        """Create FL configuration."""
        return FLConfig(
            num_clients=self.config.num_clients,
            num_rounds=self.config.num_rounds,
            local_epochs=self.config.local_epochs,
            batch_size=self.config.batch_size,
            learning_rate=self.config.learning_rate,
            aggregation_method=self.config.aggregation_method,
            unlearning_strategy=self.config.unlearning_strategy
        )
    
    def _create_unlearning_strategy(self):
        """Create unlearning strategy based on configuration."""
        if self.config.unlearning_strategy.lower() == "sisa":
            return SISAUnlearning(self.fl_config)
        elif self.config.unlearning_strategy.lower() == "gradient_negation":
            return GradientNegationUnlearning(self.fl_config, self.config.unlearning_epochs)
        elif self.config.unlearning_strategy.lower() == "knowledge_distillation":
            return KnowledgeDistillationUnlearning(self.fl_config)
        elif self.config.unlearning_strategy.lower() == "fisher_information":
            return FisherInformationUnlearning(self.fl_config)
        else:
            raise ValueError(f"Unsupported unlearning strategy: {self.config.unlearning_strategy}")
    
    def run_experiment(self) -> Dict[str, Any]:
        """Run the complete experiment."""
        self.logger.info(f"Starting experiment: {self.config.experiment_id}")
        
        self.logger.info("Loading and splitting data...")
        train_dataset, test_dataset = self.dataset_manager.load_dataset()
        
        if self.config.data_distribution.lower() == "iid":
            client_loaders = self.dataset_manager.create_iid_split(
                train_dataset, self.config.num_clients, self.config.batch_size
            )
        elif self.config.data_distribution.lower() == "non_iid":
            client_loaders = self.dataset_manager.create_non_iid_split(
                train_dataset, self.config.num_clients, self.config.batch_size,
                self.config.non_iid_alpha
            )
        elif self.config.data_distribution.lower() == "pathological":
            client_loaders = self.dataset_manager.create_pathological_split(
                train_dataset, self.config.num_clients, self.config.batch_size
            )
        else:
            raise ValueError(f"Unsupported data distribution: {self.config.data_distribution}")
        
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=self.config.batch_size, shuffle=False
        )
        
        self.logger.info("Setting up federated learning...")
        server = FedAvgServer(self.model, self.fl_config)
        
        for i, client_loader in enumerate(client_loaders):
            client_model = self._create_model()
            client = BasicFLClient(i, client_model, self.fl_config)
            client.set_local_data(client_loader)
            server.add_client(client)

        self.logger.info("Running federated training...")
        experiment = FLExperiment(self.fl_config)
        experiment.setup_experiment(server, None)
        
        fl_history = experiment.run_federated_training()
        
        fl_metrics = self.evaluator.evaluate_federated_learning(server, test_loader, fl_history)
        
        self.logger.info("Setting up unlearning experiment...")
        forget_loader, retain_loader, _ = UnlearningDataSplitter.split_for_unlearning(
            train_dataset, self.config.forget_ratio, self.config.test_ratio
        )
        
        unlearning_strategy = self._create_unlearning_strategy()
        
        self.logger.info("Running unlearning...")
        unlearning_results = experiment.run_unlearning_experiment(forget_loader, retain_loader)
        
        self.logger.info("Evaluating unlearning effectiveness...")
        unlearning_evaluation = self.evaluator.evaluate_unlearning_effectiveness(
            server.global_model, unlearning_results['unlearned_model'],
            forget_loader, retain_loader, test_loader
        )
        
        results = {
            'experiment_config': asdict(self.config),
            'fl_metrics': fl_metrics,
            'unlearning_evaluation': unlearning_evaluation,
            'fl_history': fl_history,
            'data_statistics': self.dataset_manager.get_data_statistics(client_loaders)
        }
        
        if self.config.save_results:
            self._save_results(results)
        
        if self.config.use_wandb:
            self._log_to_wandb(results)
        
        self.logger.info("Experiment completed successfully!")
        return results
    
    def _save_results(self, results: Dict[str, Any]):
        """Save experiment results."""
        results_file = self.results_dir / "results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        config_file = self.results_dir / "config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(asdict(self.config), f, default_flow_style=False)
        
        report = self.evaluator.create_evaluation_report(
            results['unlearning_evaluation'],
            str(self.results_dir / "evaluation_report.txt")
        )
        
        if results['fl_history']:
            self.evaluator.plot_training_curves(
                results['fl_history'],
                str(self.results_dir / "training_curves.png")
            )
        
        self.evaluator.plot_unlearning_comparison(
            results['unlearning_evaluation'],
            str(self.results_dir / "unlearning_comparison.png")
        )
        
        self.logger.info(f"Results saved to: {self.results_dir}")
    
    def _log_to_wandb(self, results: Dict[str, Any]):
        """Log results to Weights & Biases."""
        wandb.log({
            "fl/final_accuracy": results['fl_metrics']['final_accuracy'],
            "fl/final_loss": results['fl_metrics']['final_loss'],
            "fl/convergence_round": results['fl_metrics']['convergence_round'],
            "fl/participation_fairness": results['fl_metrics']['client_participation_fairness']
        })
        
        unlearning_metrics = results['unlearning_evaluation']['unlearning_metrics']
        wandb.log({
            "unlearning/forget_effectiveness": unlearning_metrics['forget_effectiveness'],
            "unlearning/retention_preservation": unlearning_metrics['retention_preservation'],
            "unlearning/generalization_preservation": unlearning_metrics['generalization_preservation']
        })
        
        wandb.log({
            "plots/training_curves": wandb.Image(str(self.results_dir / "training_curves.png")),
            "plots/unlearning_comparison": wandb.Image(str(self.results_dir / "unlearning_comparison.png"))
        })


def load_experiment_config(config_path: str) -> ExperimentConfig:
    """Load experiment configuration from file."""
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    return ExperimentConfig(**config_dict)


def create_default_config() -> ExperimentConfig:
    """Create a default experiment configuration."""
    return ExperimentConfig()


def run_experiment_from_config(config_path: str) -> Dict[str, Any]:
    """Run experiment from configuration file."""
    config = load_experiment_config(config_path)
    runner = ExperimentRunner(config)
    return runner.run_experiment()
