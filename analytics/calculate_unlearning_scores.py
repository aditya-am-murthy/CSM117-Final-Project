import csv
import sys
import os

# Add the current directory to the path to import analysis
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def unlearning_score(forget_acc, retain_acc, test_acc, clamp=True):
    """Calculate unlearning effectiveness score."""
    denominator = retain_acc - test_acc
    if abs(denominator) < 1e-8:
        return 0.0
    score = (forget_acc - test_acc) / denominator
    if clamp:
        score = max(-2.0, min(2.0, score))  # reasonable clipping
    return score

def safe_float(value, default=None):
    """Safely convert value to float, returning default if conversion fails."""
    if value is None or value == '':
        return default
    try:
        # Remove quotes if present
        value = str(value).strip('"').strip()
        if value == '':
            return default
        return float(value)
    except (ValueError, TypeError):
        return default

def safe_bool(value):
    """Safely convert value to boolean."""
    if value is None or value == '':
        return ''
    value = str(value).strip('"').strip().lower()
    if value in ('true', '1', 'yes'):
        return True
    elif value in ('false', '0', 'no'):
        return False
    return value

# Read all_data.csv
with open('all_data.csv', 'r') as f:
    reader = csv.DictReader(f)
    all_data = list(reader)

# Prepare output data
output_rows = []
output_headers = [
    'Model', 
    'Unlearning_Score', 
    'Test_Accuracy',
    'Generalization_Preservation',
    'Retention_Preservation',
    'Learning_Rate',
    'Forget_Ratio',
    'Pruning_Ratio',
    'Importance_Threshold',
    'Forget_Loss_Weight',
    'Prune_Classifier_Only'
]

# Process each row
for row in all_data:
    model_id = row.get('Name', '').strip('"')
    if not model_id:
        continue
    
    try:
        # Get accuracy values
        retain_acc = safe_float(row.get('unlearned/retain_accuracy'))
        forget_acc = safe_float(row.get('unlearned/forget_accuracy'))
        test_acc = safe_float(row.get('unlearned/test_accuracy'))
        
        # Skip if we don't have the required accuracy values
        if retain_acc is None or forget_acc is None or test_acc is None:
            print(f"Warning: Missing accuracy data for {model_id}, skipping...")
            continue
        
        # Calculate unlearning score
        score = unlearning_score(forget_acc, retain_acc, test_acc, clamp=True)
        
        # Get preservation metrics
        gen_preservation = safe_float(row.get('unlearning/generalization_preservation'))
        ret_preservation = safe_float(row.get('unlearning/retention_preservation'))
        
        # Get hyperparameters
        learning_rate = safe_float(row.get('learning_rate'))
        forget_ratio = safe_float(row.get('forget_ratio'))
        pruning_ratio = safe_float(row.get('pruning_ratio'))
        importance_threshold = safe_float(row.get('importance_threshold'))
        forget_loss_weight = safe_float(row.get('forget_loss_weight'))
        prune_classifier_only = safe_bool(row.get('prune_classifier_only'))
        
        # Build output row
        output_row = [
            model_id,
            score,
            test_acc,
            gen_preservation if gen_preservation is not None else '',
            ret_preservation if ret_preservation is not None else '',
            learning_rate if learning_rate is not None else '',
            forget_ratio if forget_ratio is not None else '',
            pruning_ratio if pruning_ratio is not None else '',
            importance_threshold if importance_threshold is not None else '',
            forget_loss_weight if forget_loss_weight is not None else '',
            prune_classifier_only
        ]
        
        output_rows.append(output_row)
        
    except Exception as e:
        print(f"Warning: Could not process {model_id}: {e}")
        continue

# Sort by model ID for consistency
output_rows.sort(key=lambda x: x[0])

# Write output CSV
output_file = 'unlearning_scores.csv'
with open(output_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(output_headers)
    writer.writerows(output_rows)

print(f"Unlearning scores calculated and saved to {output_file}")
print(f"Processed {len(output_rows)} model scores")
