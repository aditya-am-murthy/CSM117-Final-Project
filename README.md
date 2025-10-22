# Federated Learning Unlearning Research Project

## Overview
This project explores efficient and secure unlearning methods in federated learning settings, addressing privacy regulations like GDPR's "Right to be Forgotten." We implement and evaluate various unlearning strategies including retrain-based removal, gradient negation, and knowledge distillation.

## Research Goals
1. **Incremental Unlearning**: Develop streaming frameworks for continuous data arrivals and multiple deletion requests
2. **Utility-Privacy Balance**: Investigate optimization objectives that balance forgetting vs retention preservation
3. **Certified Unlearning**: Formalize probabilistic certification frameworks for deep networks

## Project Structure
```
├── src/                    # Source code
│   ├── fl/                # Federated learning components
│   ├── unlearning/        # Unlearning strategies
│   ├── data/              # Data handling and preprocessing
│   ├── evaluation/        # Metrics and evaluation
│   └── utils/             # Utilities and helpers
├── experiments/           # Experiment configurations and scripts
├── notebooks/             # Jupyter notebooks for analysis
├── results/              # Experiment results and plots
├── docs/                 # Documentation
├── tests/                # Unit tests
└── requirements.txt      # Dependencies
```

## Quick Start
1. Install dependencies: `pip install -r requirements.txt`
2. Run baseline experiment: `python experiments/run_baseline.py`
3. View results in `notebooks/analysis.ipynb`

## Key Features
- Modular design for easy extension
- Multiple unlearning strategies (SISA, gradient negation, knowledge distillation)
- Comprehensive evaluation metrics
- Support for CIFAR-10 and other datasets
- Configurable experiment framework

## Timeline
- Week 1: Background study and problem framing
- Week 2: Baseline setup and experimental plan
- Week 3: Implementation of proposed extension
- Week 4: Validation and benchmarking
- Week 5: Final analysis and presentation
