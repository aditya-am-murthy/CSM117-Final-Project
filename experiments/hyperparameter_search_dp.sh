#!/bin/bash

# Hyperparameter Search Script for Dynamic Pruning Unlearning
# This script tests different hyperparameter combinations and logs results

set -e  # Exit on error (but we'll handle errors in the loop)

# Configuration
BASE_CONFIG="experiments/configs/mini_batch_forgetting_dynamic_pruning.json"
SEARCH_DIR="./hyperparameter_search_results"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Create search directory
mkdir -p "${SEARCH_DIR}"

# Find existing CSV file to resume from, or create new one
EXISTING_CSV=$(ls -t "${SEARCH_DIR}"/hyperparameter_search_*.csv 2>/dev/null | head -1)
if [ -n "${EXISTING_CSV}" ] && [ -f "${EXISTING_CSV}" ]; then
    RESULTS_CSV="${EXISTING_CSV}"
    LOG_FILE="${EXISTING_CSV%.csv}.log"
    echo "Resuming from existing search: ${RESULTS_CSV}"
else
    RESULTS_CSV="${SEARCH_DIR}/hyperparameter_search_${TIMESTAMP}.csv"
    LOG_FILE="${SEARCH_DIR}/hyperparameter_search_${TIMESTAMP}.log"
    # Initialize results CSV with header
    echo "run_id,pruning_ratio,forget_loss_weight,fine_tune_epochs,prune_classifier_only,original_test_acc,gold_standard_test_acc,unlearned_test_acc,forget_effectiveness,retention_preservation,generalization_preservation,forget_accuracy_drop,retain_accuracy_drop,test_accuracy_drop,forget_gap_to_gold_standard,retain_gap_to_gold_standard,test_gap_to_gold_standard,experiment_id" > "${RESULTS_CSV}"
fi

# Hyperparameter ranges to search
PRUNING_RATIOS=(0.05 0.10 0.15 0.20)
FORGET_LOSS_WEIGHTS=(0.1 0.2 0.3 0.5)
FINE_TUNE_EPOCHS=5  # Fixed value, not searching over this
PRUNE_CLASSIFIER_ONLY=(true false)

# Function to check if a combination has already been tested
is_already_tested() {
    local pruning_ratio=$1
    local forget_loss_weight=$2
    local prune_classifier_only=$3
    
    if [ ! -f "${RESULTS_CSV}" ]; then
        return 1  # Not tested (file doesn't exist)
    fi
    
    # Use Python to check if this combination exists in CSV
    python3 << EOF
import csv
import sys

csv_file = "${RESULTS_CSV}"
pruning_ratio = "${pruning_ratio}"
forget_loss_weight = "${forget_loss_weight}"
prune_classifier_only = "${prune_classifier_only}"

try:
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if (row.get('pruning_ratio') == pruning_ratio and
                row.get('forget_loss_weight') == forget_loss_weight and
                row.get('prune_classifier_only') == prune_classifier_only):
                # Check if it's a valid result (not ERROR or FAILED)
                forget_eff = row.get('forget_effectiveness', '')
                if forget_eff not in ['ERROR', 'FAILED', 'N/A', '']:
                    sys.exit(0)  # Found valid result
                else:
                    sys.exit(1)  # Found but failed/error, should retry
    sys.exit(1)  # Not found
except Exception as e:
    sys.exit(1)  # Error reading file, assume not tested
EOF
}

# Counter for runs
RUN_COUNT=0
TOTAL_RUNS=$((${#PRUNING_RATIOS[@]} * ${#FORGET_LOSS_WEIGHTS[@]} * ${#PRUNE_CLASSIFIER_ONLY[@]}))

# Count already completed runs
COMPLETED_RUNS=0
if [ -f "${RESULTS_CSV}" ]; then
    COMPLETED_RUNS=$(tail -n +2 "${RESULTS_CSV}" 2>/dev/null | wc -l | tr -d ' ')
fi

echo "============================================================"
echo "Hyperparameter Search for Dynamic Pruning Unlearning"
echo "============================================================"
echo "Total combinations to test: ${TOTAL_RUNS}"
if [ ${COMPLETED_RUNS} -gt 0 ]; then
    echo "Already completed: ${COMPLETED_RUNS}"
    echo "Remaining: $((TOTAL_RUNS - COMPLETED_RUNS))"
fi
echo "Results will be saved to: ${RESULTS_CSV}"
echo "Log file: ${LOG_FILE}"
echo "============================================================"
echo ""

# Function to extract metrics from experiment_log.json
extract_metrics() {
    local exp_id=$1
    local log_file="./results/${exp_id}/experiment_log.json"
    
    if [ ! -f "${log_file}" ]; then
        echo "ERROR: Log file not found: ${log_file}" >&2
        return 1
    fi
    
    # Use Python to parse JSON (more reliable than jq)
    python3 << EOF
import json
import sys

try:
    with open("${log_file}", 'r') as f:
        data = json.load(f)
    
    metrics = data.get('metrics', {})
    
    # Extract all metrics
    results = {
        'original_test_acc': metrics.get('original_test_accuracy', 'N/A'),
        'gold_standard_test_acc': metrics.get('gold_standard_test_accuracy', 'N/A'),
        'unlearned_test_acc': metrics.get('unlearned_test_accuracy', 'N/A'),
        'forget_effectiveness': metrics.get('forget_effectiveness', 'N/A'),
        'retention_preservation': metrics.get('retention_preservation', 'N/A'),
        'generalization_preservation': metrics.get('generalization_preservation', 'N/A'),
        'forget_accuracy_drop': metrics.get('forget_accuracy_drop', 'N/A'),
        'retain_accuracy_drop': metrics.get('retain_accuracy_drop', 'N/A'),
        'test_accuracy_drop': metrics.get('test_accuracy_drop', 'N/A'),
        'forget_gap_to_gold_standard': metrics.get('forget_gap_to_gold_standard', 'N/A'),
        'retain_gap_to_gold_standard': metrics.get('retain_gap_to_gold_standard', 'N/A'),
        'test_gap_to_gold_standard': metrics.get('test_gap_to_gold_standard', 'N/A')
    }
    
    # Print as comma-separated values
    print(','.join([str(v) for v in results.values()]))
except Exception as e:
    print(f"ERROR: {e}", file=sys.stderr)
    sys.exit(1)
EOF
}

# Function to create temporary config file
create_temp_config() {
    local temp_config=$1
    local pruning_ratio=$2
    local forget_loss_weight=$3
    local prune_classifier_only=$4
    local run_count=$5
    
    # Convert bash boolean to Python boolean
    local python_bool="True"
    if [ "${prune_classifier_only}" = "false" ]; then
        python_bool="False"
    fi
    
    # Use Python to modify JSON
    python3 << EOF
import json
import sys
import datetime

with open("${BASE_CONFIG}", 'r') as f:
    config = json.load(f)

config['pruning_ratio'] = ${pruning_ratio}
config['forget_loss_weight'] = ${forget_loss_weight}
config['fine_tune_epochs'] = ${FINE_TUNE_EPOCHS}  # Fixed value
config['prune_classifier_only'] = ${python_bool}
config['batch_size'] = 23  # Always use batch size 23
config['device'] = 'cuda:3'  # Always use GPU 3

# Add a unique experiment_id for this run
timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
config['experiment_id'] = f"hyperparam_search_{timestamp}_{${run_count}}"

with open("${temp_config}", 'w') as f:
    json.dump(config, f, indent=2)

print(config['experiment_id'])
EOF
}

# Main search loop
for pruning_ratio in "${PRUNING_RATIOS[@]}"; do
    for forget_loss_weight in "${FORGET_LOSS_WEIGHTS[@]}"; do
        for prune_classifier_only in "${PRUNE_CLASSIFIER_ONLY[@]}"; do
            RUN_COUNT=$((RUN_COUNT + 1))
            
            # Check if this combination has already been tested
            if is_already_tested "${pruning_ratio}" "${forget_loss_weight}" "${prune_classifier_only}"; then
                echo "[${RUN_COUNT}/${TOTAL_RUNS}] ⏭️  Skipping (already completed): pruning_ratio=${pruning_ratio}, forget_loss_weight=${forget_loss_weight}, prune_classifier_only=${prune_classifier_only}"
                echo "[${RUN_COUNT}/${TOTAL_RUNS}] ⏭️  Skipped (already completed): pruning_ratio=${pruning_ratio}, forget_loss_weight=${forget_loss_weight}, prune_classifier_only=${prune_classifier_only} - $(date)" | tee -a "${LOG_FILE}"
                continue
            fi
            
            echo "[${RUN_COUNT}/${TOTAL_RUNS}] Testing: pruning_ratio=${pruning_ratio}, forget_loss_weight=${forget_loss_weight}, fine_tune_epochs=${FINE_TUNE_EPOCHS} (fixed), prune_classifier_only=${prune_classifier_only}"
            echo "[${RUN_COUNT}/${TOTAL_RUNS}] $(date)" | tee -a "${LOG_FILE}"
            
            # Create temporary config
            TEMP_CONFIG=$(mktemp).json
            EXPERIMENT_ID=$(create_temp_config "${TEMP_CONFIG}" "${pruning_ratio}" "${forget_loss_weight}" "${prune_classifier_only}" "${RUN_COUNT}")
                
            echo "  Experiment ID: ${EXPERIMENT_ID}" | tee -a "${LOG_FILE}"
            
            # Run experiment (don't exit on error, just log it)
            echo "  Running experiment..." | tee -a "${LOG_FILE}"
            set +e  # Temporarily disable exit on error
            if python3 experiments/run_experiments.py --config "${TEMP_CONFIG}" --verbose 2>&1 | tee -a "${LOG_FILE}"; then
                set -e  # Re-enable exit on error
                echo "  ✓ Experiment completed successfully" | tee -a "${LOG_FILE}"
                
                # Extract metrics
                echo "  Extracting metrics..." | tee -a "${LOG_FILE}"
                METRICS=$(extract_metrics "${EXPERIMENT_ID}" 2>&1)
                
                if [ $? -eq 0 ] && [ -n "${METRICS}" ]; then
                    # Append to CSV
                    echo "${RUN_COUNT},${pruning_ratio},${forget_loss_weight},${FINE_TUNE_EPOCHS},${prune_classifier_only},${METRICS},${EXPERIMENT_ID}" >> "${RESULTS_CSV}"
                    echo "  ✓ Metrics extracted and logged" | tee -a "${LOG_FILE}"
                else
                    echo "  ✗ Failed to extract metrics: ${METRICS}" | tee -a "${LOG_FILE}"
                    echo "${RUN_COUNT},${pruning_ratio},${forget_loss_weight},${FINE_TUNE_EPOCHS},${prune_classifier_only},ERROR,ERROR,ERROR,ERROR,ERROR,ERROR,ERROR,ERROR,ERROR,ERROR,ERROR,ERROR,${EXPERIMENT_ID}" >> "${RESULTS_CSV}"
                fi
            else
                set -e  # Re-enable exit on error
                echo "  ✗ Experiment failed" | tee -a "${LOG_FILE}"
                echo "${RUN_COUNT},${pruning_ratio},${forget_loss_weight},${FINE_TUNE_EPOCHS},${prune_classifier_only},FAILED,FAILED,FAILED,FAILED,FAILED,FAILED,FAILED,FAILED,FAILED,FAILED,FAILED,FAILED,${EXPERIMENT_ID}" >> "${RESULTS_CSV}"
            fi
            
            # Clean up temp config
            rm -f "${TEMP_CONFIG}"
            
            # Flush output to ensure results are saved immediately
            sync
            
            echo "" | tee -a "${LOG_FILE}"
        done
    done
done

echo ""
echo "============================================================"
echo "Hyperparameter Search Complete!"
echo "============================================================"
echo "Total runs attempted: ${RUN_COUNT}"
if [ -f "${RESULTS_CSV}" ]; then
    COMPLETED=$(tail -n +2 "${RESULTS_CSV}" 2>/dev/null | wc -l | tr -d ' ')
    echo "Completed runs: ${COMPLETED}"
    echo "Remaining: $((TOTAL_RUNS - COMPLETED))"
fi
echo "Results CSV: ${RESULTS_CSV}"
echo "Log file: ${LOG_FILE}"
echo ""
echo "To resume this search, simply run this script again - it will"
echo "automatically detect and skip already completed combinations."
echo ""
echo "Top 10 configurations by forget_effectiveness:"
echo ""

# Use Python to analyze and display top results
python3 << EOF
import csv
import sys

csv_file = "${RESULTS_CSV}"

try:
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    # Filter out rows with errors
    valid_rows = [r for r in rows if r.get('forget_effectiveness', '') not in ['N/A', 'ERROR', 'FAILED', '']]
    
    # Sort by forget_effectiveness (descending)
    try:
        valid_rows.sort(key=lambda x: float(x['forget_effectiveness']) if x['forget_effectiveness'] != 'N/A' else -999, reverse=True)
    except (ValueError, KeyError):
        print("Warning: Could not sort by forget_effectiveness")
    
    print(f"{'Rank':<6} {'Pruning':<8} {'Forget Loss':<12} {'Class Only':<12} {'Forget Eff':<12} {'Retention':<12} {'Test Acc':<10}")
    print("-" * 80)
    
    for i, row in enumerate(valid_rows[:10], 1):
        pruning = row.get('pruning_ratio', 'N/A')
        forget_loss = row.get('forget_loss_weight', 'N/A')
        class_only = row.get('prune_classifier_only', 'N/A')
        forget_eff = row.get('forget_effectiveness', 'N/A')
        retention = row.get('retention_preservation', 'N/A')
        test_acc = row.get('unlearned_test_acc', 'N/A')
        
        print(f"{i:<6} {pruning:<8} {forget_loss:<12} {class_only:<12} {forget_eff:<12} {retention:<12} {test_acc:<10}")
    
    if len(valid_rows) == 0:
        print("No valid results found. Check the log file for errors.")
        
except Exception as e:
    print(f"Error analyzing results: {e}", file=sys.stderr)
    sys.exit(1)
EOF

echo ""
echo "============================================================"

