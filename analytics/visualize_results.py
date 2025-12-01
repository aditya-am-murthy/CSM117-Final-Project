import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)

# Read the data
df_scores = pd.read_csv('unlearning_scores.csv')
df_all = pd.read_csv('all_data.csv')

# Merge to get forget accuracy
df = df_scores.merge(
    df_all[['Name', 'unlearned/forget_accuracy']], 
    left_on='Model', 
    right_on='Name', 
    how='left'
)

# Calculate forget metric (test - forget accuracy)
df['Forget_Metric'] = df['Test_Accuracy'] - df['unlearned/forget_accuracy']

# Convert Prune_Classifier_Only to numeric for better visualization
# Map True/False strings to 0/1
df['Prune_Classifier_Only'] = df['Prune_Classifier_Only'].map({True: 1, False: 0, 'True': 1, 'False': 0, 'true': 1, 'false': 0})

# Create figure with subplots
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
fig.suptitle('Unlearning Analysis: Score and Forget Metric vs Hyperparameters', 
             fontsize=16, fontweight='bold', y=1.02)

# X-axis variables to plot
x_vars = ['Forget_Ratio', 'Pruning_Ratio', 'Forget_Loss_Weight', 'Prune_Classifier_Only']
x_labels = ['Forget Ratio', 'Pruning Ratio', 'Forget Loss Weight', 'Prune Classifier Only']

# Plot Unlearning Score
for idx, (x_var, x_label) in enumerate(zip(x_vars, x_labels)):
    ax = axes[0, idx]
    
    if x_var == 'Prune_Classifier_Only':
        # For categorical variable, use scatter with different colors
        for val in sorted(df[x_var].dropna().unique()):
            mask = df[x_var] == val
            label = 'True' if val == 1 else 'False'
            ax.scatter(df.loc[mask, x_var], df.loc[mask, 'Unlearning_Score'], 
                      alpha=0.7, s=100, label=label)
        ax.set_xlabel(x_label, fontsize=11)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['False', 'True'])
        ax.legend()
    else:
        ax.scatter(df[x_var], df['Unlearning_Score'], alpha=0.7, s=100, c='steelblue')
        ax.set_xlabel(x_label, fontsize=11)
    
    ax.set_ylabel('Unlearning Score', fontsize=11)
    ax.set_title(f'Unlearning Score vs {x_label}', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add trend line for continuous variables (only if there's variation)
    if x_var != 'Prune_Classifier_Only' and df[x_var].nunique() > 1:
        try:
            z = np.polyfit(df[x_var].dropna(), df['Unlearning_Score'].dropna(), 1)
            p = np.poly1d(z)
            x_line = np.linspace(df[x_var].min(), df[x_var].max(), 100)
            ax.plot(x_line, p(x_line), "r--", alpha=0.5, linewidth=2, label='Trend')
            if idx == 0:
                ax.legend()
        except:
            pass

# Plot Forget Metric (Test - Forget Accuracy)
for idx, (x_var, x_label) in enumerate(zip(x_vars, x_labels)):
    ax = axes[1, idx]
    
    if x_var == 'Prune_Classifier_Only':
        # For categorical variable, use scatter with different colors
        for val in sorted(df[x_var].dropna().unique()):
            mask = df[x_var] == val
            label = 'True' if val == 1 else 'False'
            ax.scatter(df.loc[mask, x_var], df.loc[mask, 'Forget_Metric'], 
                      alpha=0.7, s=100, label=label)
        ax.set_xlabel(x_label, fontsize=11)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['False', 'True'])
        ax.legend()
    else:
        ax.scatter(df[x_var], df['Forget_Metric'], alpha=0.7, s=100, c='coral')
        ax.set_xlabel(x_label, fontsize=11)
    
    ax.set_ylabel('Forget Metric (Test - Forget Accuracy)', fontsize=11)
    ax.set_title(f'Forget Metric vs {x_label}', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add trend line for continuous variables (only if there's variation)
    if x_var != 'Prune_Classifier_Only' and df[x_var].nunique() > 1:
        try:
            z = np.polyfit(df[x_var].dropna(), df['Forget_Metric'].dropna(), 1)
            p = np.poly1d(z)
            x_line = np.linspace(df[x_var].min(), df[x_var].max(), 100)
            ax.plot(x_line, p(x_line), "r--", alpha=0.5, linewidth=2, label='Trend')
            if idx == 0:
                ax.legend()
        except:
            pass

plt.tight_layout()
plt.savefig('unlearning_analysis.png', dpi=300, bbox_inches='tight')
print("Visualization saved as 'unlearning_analysis.png'")

# Identify top 3 models
# For unlearning, higher score is better (closer to 0 or positive)
# For forget metric, higher is better (larger gap between test and forget accuracy)
# We'll rank by a combination of both metrics

# Normalize metrics for ranking
df['Unlearning_Score_Norm'] = (df['Unlearning_Score'] - df['Unlearning_Score'].min()) / (df['Unlearning_Score'].max() - df['Unlearning_Score'].min())
df['Forget_Metric_Norm'] = (df['Forget_Metric'] - df['Forget_Metric'].min()) / (df['Forget_Metric'].max() - df['Forget_Metric'].min())

# Combined score (weighted average - higher unlearning score and higher forget metric are better)
df['Combined_Score'] = 0.5 * df['Unlearning_Score_Norm'] + 0.5 * df['Forget_Metric_Norm']

# Sort by combined score
df_sorted = df.sort_values('Combined_Score', ascending=False)

print("\n" + "="*80)
print("TOP 3 MODELS")
print("="*80)

for i, (idx, row) in enumerate(df_sorted.head(3).iterrows(), 1):
    print(f"\n{i}. {row['Model']}")
    print(f"   Unlearning Score: {row['Unlearning_Score']:.4f}")
    print(f"   Test Accuracy: {row['Test_Accuracy']:.4f}")
    forget_acc = row.get('unlearned/forget_accuracy', 'N/A')
    if forget_acc != 'N/A' and pd.notna(forget_acc):
        print(f"   Forget Accuracy: {float(forget_acc):.4f}")
    else:
        print(f"   Forget Accuracy: N/A")
    print(f"   Forget Metric (Test - Forget): {row['Forget_Metric']:.4f}")
    print(f"   Generalization Preservation: {row['Generalization_Preservation']:.4f}")
    print(f"   Retention Preservation: {row['Retention_Preservation']:.4f}")
    print(f"   Hyperparameters:")
    print(f"     - Learning Rate: {row['Learning_Rate']}")
    print(f"     - Forget Ratio: {row['Forget_Ratio']}")
    print(f"     - Pruning Ratio: {row['Pruning_Ratio']}")
    print(f"     - Importance Threshold: {row['Importance_Threshold']}")
    print(f"     - Forget Loss Weight: {row['Forget_Loss_Weight']}")
    prune_val = row['Prune_Classifier_Only']
    prune_str = 'True' if prune_val == 1 or str(prune_val).lower() == 'true' else 'False'
    print(f"     - Prune Classifier Only: {prune_str}")

# Interpretation
print("\n" + "="*80)
print("INTERPRETATION OF RESULTS")
print("="*80)

print("\n1. UNLEARNING SCORE ANALYSIS:")
print("   - Higher unlearning scores (closer to 0 or positive) indicate better unlearning effectiveness")
print(f"   - Score range: {df['Unlearning_Score'].min():.4f} to {df['Unlearning_Score'].max():.4f}")
print(f"   - Mean score: {df['Unlearning_Score'].mean():.4f}")

print("\n2. FORGET METRIC ANALYSIS (Test - Forget Accuracy):")
print("   - Higher values indicate larger gap between test and forget accuracy (better forgetting)")
print(f"   - Metric range: {df['Forget_Metric'].min():.4f} to {df['Forget_Metric'].max():.4f}")
print(f"   - Mean metric: {df['Forget_Metric'].mean():.4f}")

print("\n3. HYPERPARAMETER EFFECTS:")

# Analyze each hyperparameter
for x_var, x_label in zip(x_vars, x_labels):
    if x_var != 'Prune_Classifier_Only':
        if df[x_var].nunique() > 1:
            corr_score = df[x_var].corr(df['Unlearning_Score'])
            corr_forget = df[x_var].corr(df['Forget_Metric'])
            print(f"\n   {x_label}:")
            print(f"     - Correlation with Unlearning Score: {corr_score:.4f}")
            print(f"     - Correlation with Forget Metric: {corr_forget:.4f}")
        else:
            print(f"\n   {x_label}:")
            print(f"     - All models use the same value: {df[x_var].iloc[0]}")
            print(f"     - Cannot compute correlation (no variation)")
    else:
        grouped = df.groupby('Prune_Classifier_Only').agg({
            'Unlearning_Score': 'mean',
            'Forget_Metric': 'mean'
        })
        print(f"\n   {x_label}:")
        for val in sorted(grouped.index):
            val_label = 'True' if val == 1 else 'False'
            print(f"     - {val_label}: Mean Unlearning Score = {grouped.loc[val, 'Unlearning_Score']:.4f}, "
                  f"Mean Forget Metric = {grouped.loc[val, 'Forget_Metric']:.4f}")

print("\n4. KEY INSIGHTS:")
print("   - Models with better unlearning typically show:")
print("     * Higher forget metrics (larger test-forget gap)")
print("     * Better retention of generalization and retention preservation")
print("   - The top 3 models balance unlearning effectiveness with model performance")

# Save detailed results
df_sorted[['Model', 'Unlearning_Score', 'Test_Accuracy', 'Forget_Metric', 
          'Generalization_Preservation', 'Retention_Preservation',
          'Learning_Rate', 'Forget_Ratio', 'Pruning_Ratio', 
          'Importance_Threshold', 'Forget_Loss_Weight', 'Prune_Classifier_Only',
          'Combined_Score']].to_csv('model_rankings.csv', index=False)
print("\nDetailed rankings saved to 'model_rankings.csv'")

