"""
Analyze and visualize hyperparameter search results from CSV.

This script loads the CSV results and provides analysis, visualizations,
and identifies the best configurations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse


def load_results(csv_path):
    """Load results from CSV file."""
    print(f"Loading results from: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"✓ Loaded {len(df)} runs")
    
    # Filter successful runs
    df_success = df[df['status'] == 'success'].copy()
    print(f"✓ {len(df_success)} successful runs")
    print(f"✗ {len(df) - len(df_success)} failed runs")
    
    return df, df_success


def print_summary_statistics(df):
    """Print summary statistics."""
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    metrics = [
        'unlearning_effectiveness',
        'retention_preservation', 
        'forget_accuracy_improvement',
        'retain_accuracy_change'
    ]
    
    for metric in metrics:
        if metric in df.columns:
            print(f"\n{metric.replace('_', ' ').title()}:")
            print(f"  Mean:   {df[metric].mean():.4f}")
            print(f"  Std:    {df[metric].std():.4f}")
            print(f"  Min:    {df[metric].min():.4f}")
            print(f"  Max:    {df[metric].max():.4f}")
            print(f"  Median: {df[metric].median():.4f}")


def show_best_configurations(df, n=10):
    """Show top N configurations by different metrics."""
    print("\n" + "="*80)
    print(f"TOP {n} CONFIGURATIONS")
    print("="*80)
    
    # Add balanced score
    df['balanced_score'] = (
        0.5 * df['unlearning_effectiveness'] + 
        0.5 * df['retention_preservation']
    )
    
    # Top by unlearning effectiveness
    print("\n--- By Unlearning Effectiveness ---")
    top_unlearn = df.nlargest(n, 'unlearning_effectiveness')
    print(top_unlearn[[
        'run_id', 'pruning_ratio', 'forget_weight', 'pareto_steps',
        'unlearning_effectiveness', 'retention_preservation'
    ]].to_string(index=False))
    
    # Top by retention preservation
    print("\n--- By Retention Preservation ---")
    top_retain = df.nlargest(n, 'retention_preservation')
    print(top_retain[[
        'run_id', 'pruning_ratio', 'forget_weight', 'pareto_steps',
        'unlearning_effectiveness', 'retention_preservation'
    ]].to_string(index=False))
    
    # Top by balanced score
    print("\n--- By Balanced Score ---")
    top_balanced = df.nlargest(n, 'balanced_score')
    print(top_balanced[[
        'run_id', 'pruning_ratio', 'forget_weight', 'pareto_steps',
        'balanced_score', 'unlearning_effectiveness', 'retention_preservation'
    ]].to_string(index=False))
    
    return top_balanced.iloc[0]


def analyze_hyperparameter_impact(df):
    """Analyze impact of each hyperparameter."""
    print("\n" + "="*80)
    print("HYPERPARAMETER IMPACT ANALYSIS")
    print("="*80)
    
    hyperparams = [
        'pruning_ratio', 'importance_threshold', 'pruning_iterations',
        'forget_weight', 'retention_weight', 'pareto_steps',
        'phase1_epochs', 'phase2_epochs', 'refinement_epochs',
        'learning_rate'
    ]
    
    df['balanced_score'] = (
        0.5 * df['unlearning_effectiveness'] + 
        0.5 * df['retention_preservation']
    )
    
    for param in hyperparams:
        if param in df.columns:
            grouped = df.groupby(param)['balanced_score'].agg(['mean', 'std', 'count'])
            print(f"\n{param}:")
            print(grouped.to_string())


def create_visualizations(df, output_dir='./plots'):
    """Create visualizations of results."""
    Path(output_dir).mkdir(exist_ok=True)
    print(f"\nCreating visualizations in {output_dir}/...")
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 8)
    
    # 1. Scatter plot: Unlearning vs Retention
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        df['unlearning_effectiveness'], 
        df['retention_preservation'],
        c=df['pruning_ratio'],
        s=100,
        alpha=0.6,
        cmap='viridis'
    )
    plt.colorbar(scatter, label='Pruning Ratio')
    plt.xlabel('Unlearning Effectiveness', fontsize=12)
    plt.ylabel('Retention Preservation', fontsize=12)
    plt.title('Unlearning Effectiveness vs Retention Preservation', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # Add diagonal line (equal performance)
    max_val = max(df['unlearning_effectiveness'].max(), df['retention_preservation'].max())
    plt.plot([0, max_val], [0, max_val], 'r--', alpha=0.5, label='Equal Performance')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/unlearning_vs_retention.png', dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: unlearning_vs_retention.png")
    plt.close()
    
    # 2. Hyperparameter distributions
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    key_params = [
        'pruning_ratio', 'forget_weight', 'pareto_steps',
        'learning_rate', 'phase2_epochs', 'pruning_iterations'
    ]
    
    df['balanced_score'] = (
        0.5 * df['unlearning_effectiveness'] + 
        0.5 * df['retention_preservation']
    )
    
    for idx, param in enumerate(key_params):
        if param in df.columns and idx < len(axes):
            grouped = df.groupby(param)['balanced_score'].mean().sort_index()
            axes[idx].bar(range(len(grouped)), grouped.values, alpha=0.7)
            axes[idx].set_xticks(range(len(grouped)))
            axes[idx].set_xticklabels([f'{x:.3f}' for x in grouped.index], rotation=45)
            axes[idx].set_xlabel(param.replace('_', ' ').title())
            axes[idx].set_ylabel('Mean Balanced Score')
            axes[idx].set_title(f'Impact of {param.replace("_", " ").title()}')
            axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/hyperparameter_impact.png', dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: hyperparameter_impact.png")
    plt.close()
    
    # 3. Performance distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    axes[0].hist(df['unlearning_effectiveness'], bins=20, alpha=0.7, color='#ff6b6b', edgecolor='black')
    axes[0].axvline(df['unlearning_effectiveness'].mean(), color='red', linestyle='--', linewidth=2, label='Mean')
    axes[0].axvline(df['unlearning_effectiveness'].median(), color='blue', linestyle='--', linewidth=2, label='Median')
    axes[0].set_xlabel('Unlearning Effectiveness', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title('Distribution of Unlearning Effectiveness', fontsize=14)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].hist(df['retention_preservation'], bins=20, alpha=0.7, color='#51cf66', edgecolor='black')
    axes[1].axvline(df['retention_preservation'].mean(), color='red', linestyle='--', linewidth=2, label='Mean')
    axes[1].axvline(df['retention_preservation'].median(), color='blue', linestyle='--', linewidth=2, label='Median')
    axes[1].set_xlabel('Retention Preservation', fontsize=12)
    axes[1].set_ylabel('Frequency', fontsize=12)
    axes[1].set_title('Distribution of Retention Preservation', fontsize=14)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/performance_distributions.png', dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: performance_distributions.png")
    plt.close()
    
    # 4. Phase progression
    if all(col in df.columns for col in ['phase1_forget_acc', 'phase2_forget_acc', 'phase3_forget_acc']):
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        phases = ['Phase 1', 'Phase 2', 'Phase 3']
        forget_means = [
            df['phase1_forget_acc'].mean(),
            df['phase2_forget_acc'].mean(),
            df['phase3_forget_acc'].mean()
        ]
        retain_means = [
            df['phase1_retain_acc'].mean(),
            df['phase2_retain_acc'].mean(),
            df['phase3_retain_acc'].mean()
        ]
        
        x = np.arange(len(phases))
        width = 0.35
        
        axes[0].bar(x - width/2, forget_means, width, label='Forget Accuracy', alpha=0.8, color='#ff6b6b')
        axes[0].bar(x + width/2, retain_means, width, label='Retain Accuracy', alpha=0.8, color='#51cf66')
        axes[0].set_xlabel('Phase', fontsize=12)
        axes[0].set_ylabel('Accuracy', fontsize=12)
        axes[0].set_title('Average Accuracy Across Phases', fontsize=14)
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(phases)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot individual runs
        for idx in df.index[:min(10, len(df))]:
            forget_prog = [
                df.loc[idx, 'phase1_forget_acc'],
                df.loc[idx, 'phase2_forget_acc'],
                df.loc[idx, 'phase3_forget_acc']
            ]
            retain_prog = [
                df.loc[idx, 'phase1_retain_acc'],
                df.loc[idx, 'phase2_retain_acc'],
                df.loc[idx, 'phase3_retain_acc']
            ]
            axes[1].plot(phases, forget_prog, 'o-', alpha=0.3, color='#ff6b6b')
            axes[1].plot(phases, retain_prog, 's-', alpha=0.3, color='#51cf66')
        
        axes[1].set_xlabel('Phase', fontsize=12)
        axes[1].set_ylabel('Accuracy', fontsize=12)
        axes[1].set_title('Phase Progression (First 10 Runs)', fontsize=14)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/phase_progression.png', dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: phase_progression.png")
        plt.close()
    
    # 5. Correlation heatmap
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    key_cols = [col for col in numeric_cols if col in [
        'pruning_ratio', 'forget_weight', 'pareto_steps', 'learning_rate',
        'unlearning_effectiveness', 'retention_preservation', 'balanced_score'
    ]]
    
    if len(key_cols) > 2:
        plt.figure(figsize=(10, 8))
        corr = df[key_cols].corr()
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                   square=True, linewidths=1)
        plt.title('Correlation Heatmap', fontsize=14)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/correlation_heatmap.png', dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: correlation_heatmap.png")
        plt.close()
    
    print(f"\n✓ All visualizations saved to {output_dir}/")


def export_best_config(best_config, output_path='best_config.json'):
    """Export best configuration to JSON."""
    import json
    
    config_dict = best_config.to_dict()
    
    with open(output_path, 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    print(f"\n✓ Best configuration saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Analyze hyperparameter search results')
    parser.add_argument('--csv_file', type=str, default='/home/carolinewei/CSM117-Final-Project/results/hyperparam_search_results_20251130_220523.csv')
    parser.add_argument('--output-dir', type=str, default='./plots',
                       help='Directory to save plots')
    parser.add_argument('--top-n', type=int, default=10,
                       help='Number of top configurations to show')
    parser.add_argument('--export-best', action='store_true',
                       help='Export best configuration to JSON')
    
    args = parser.parse_args()
    
    # Load results
    df_all, df_success = load_results(args.csv_file)
    
    if len(df_success) == 0:
        print("\n❌ No successful runs found in CSV!")
        return
    
    # Print summary
    print_summary_statistics(df_success)
    
    # Show best configurations
    best_config = show_best_configurations(df_success, n=args.top_n)
    
    # Analyze hyperparameter impact
    analyze_hyperparameter_impact(df_success)
    
    # Create visualizations
    create_visualizations(df_success, output_dir=args.output_dir)
    
    # Export best config if requested
    if args.export_best:
        export_best_config(best_config, output_path=f'{args.output_dir}/best_config.json')
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print(f"Results from: {args.csv_file}")
    print(f"Plots saved to: {args.output_dir}/")
    print(f"Total runs analyzed: {len(df_success)}")


if __name__ == "__main__":
    main()