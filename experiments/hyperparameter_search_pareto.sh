#!/bin/bash

# =============================================================================
# Hyperparameter Search for Hybrid Pareto-Pruning Unlearning (FIXED & TESTED)
# =============================================================================

set -euo pipefail

# -------------------------- Configuration ------------------------------------
BASE_CONFIG="experiments/configs/hybrid_pareto_pruning_base.json"
SEARCH_DIR="./hyperparameter_search_hybrid_pareto"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

mkdir -p "${SEARCH_DIR}"

# Resume from latest CSV if exists
EXISTING_CSV=$(ls -t "${SEARCH_DIR}"/hybrid_pareto_search_*.csv 2>/dev/null | head -1 || true)
if [[ -n "${EXISTING_CSV}" && -f "${EXISTING_CSV}" ]]; then
    RESULTS_CSV="${EXISTING_CSV}"
    LOG_FILE="${EXISTING_CSV%.csv}.log"
    echo "Resuming from existing search: ${RESULTS_CSV}"
else
    RESULTS_CSV="${SEARCH_DIR}/hybrid_pareto_search_${TIMESTAMP}.csv"
    LOG_FILE="${SEARCH_DIR}/hybrid_pareto_search_${TIMESTAMP}.log"
    echo "run_id,pruning_ratio,forget_weight,phase1_epochs,phase2_epochs,refinement_epochs,use_gradient_masking,forget_accuracy,retain_accuracy,unlearning_effectiveness,retention_preservation,pruning_ratio_actual,pareto_distance_to_ideal,final_forget_weight,final_retain,experiment_id" > "${RESULTS_CSV}"
fi

# ---------------------- Hyperparameter Space --------------------------------
PRUNING_RATIOS=(0.10 0.15 0.20 0.25)
FORGET_WEIGHTS=(0.3 0.4 0.5 0.6 0.7)
PHASE1_EPOCHS=(3 5)
PHASE2_EPOCHS=(5 8)
REFINEMENT_EPOCHS=(3 5)
USE_GRADIENT_MASKING=("true" "false")

DEVICE="cuda:3"
BATCH_SIZE=32

# -------------------------- Helpers ------------------------------------------
is_already_tested() {
    local pr=$1 fw=$2 p1=$3 p2=$4 re=$5 gm=$6
    [[ ! -f "${RESULTS_CSV}" ]] && return 1
    python3 - <<'PY' "$pr" "$fw" "$p1" "$p2" "$re" "$gm" "${RESULTS_CSV}"
import csv, sys
pr, fw, p1, p2, re, gm, csvfile = sys.argv[1:]
pr, fw = float(pr), float(fw)
p1, p2, re = int(p1), int(p2), int(re)
with open(csvfile) as f:
    for row in csv.DictReader(f):
        if (abs(float(row['pruning_ratio']) - pr) < 1e-6 and
            abs(float(row['forget_weight']) - fw) < 1e-6 and
            int(row['phase1_epochs']) == p1 and
            int(row['phase2_epochs']) == p2 and
            int(row['refinement_epochs']) == re and
            row['use_gradient_masking'] == gm and
            row['unlearning_effectiveness'] not in ['ERROR','FAILED','N/A','']):
            sys.exit(0)
sys.exit(1)
PY
}

create_temp_config() {
    local temp=$1
    local pr=$2 fw=$3 p1=$4 p2=$5 re=$6 gm=$7 run_id=$8

    python3 - <<PY "${temp}" "${pr}" "${fw}" "${p1}" "${p2}" "${re}" "${gm}" "${run_id}"
import json, datetime
temp, pr, fw, p1, p2, re, gm, run_id = __import__('sys').argv')[1:]

with open("${BASE_CONFIG}") as f:
    cfg = json.load(f)

cfg.update({
    "unlearning_strategy": "HybridParetoePruningUnlearning",
    "pruning_ratio": float(pr),
    "forget_weight": float(fw),
    "retention_weight": round(1.0 - float(fw), 3),
    "phase1_epochs": int(p1),
    "phase2_epochs": int(p2),
    "refinement_epochs": int(re),
    "use_gradient_masking": gm.lower() == "true",
    "device": "${DEVICE}",
    "batch_size": ${BATCH_SIZE},
    "experiment_id": f"hybrid_hp_{run_id}_{datetime.datetime.now():%H%M%S}"
})

with open(temp, "w") as f:
    json.dump(cfg, f, indent=2)

print(cfg["experiment_id"])
PY
}

extract_metrics() {
    local exp_id=$1
    local log="./results/${exp_id}/experiment_log.json"
    [[ -f "${log}" ]] || { echo "ERROR"; return 1; }

    python3 - <<PY "${log}"
import json, sys, math
try:
    data = json.load(open(sys.argv[1]))
    m = data.get("metrics", {})

    forget_acc = m.get("forget_accuracy", 0.0)
    retain_acc = m.get("retain_accuracy", 1.0)
    dist = math.sqrt((1.0 - forget_acc)**2 + (1.0 - retain_acc)**2)

    print(f"{forget_acc},{retain_acc},{m.get('unlearning_effectiveness', 'N/A')}," +
          f"{m.get('retention_preservation', 'N/A')},{m.get('pruning_ratio', 'N/A')}," +
          f"{dist:.6f},{m.get('final_forget_weight', 'N/A')}," +
          f"{m.get('final_retention_weight', 'N/A')}")
except Exception as e:
    print("ERROR,ERROR,ERROR,ERROR,ERROR,ERROR,ERROR,ERROR", file=sys.stderr)
    print("ERROR")
PY
}

# ------------------------------ Main Loop -------------------------------------
RUN_COUNT=0
TOTAL=$(( ${#PRUNING_RATIOS[@]} * ${#FORGET_WEIGHTS[@]} * ${#PHASE1_EPOCHS[@]} * \
           ${#PHASE2_EPOCHS[@]} * ${#REFINEMENT_EPOCHS[@]} * ${#USE_GRADIENT_MASKING[@]} ))

echo "======================================================================"
echo "Hybrid Pareto-Pruning Unlearning – Hyperparameter Search"
echo "======================================================================"
echo "Total combinations: $TOTAL"
echo "Results → $RESULTS_CSV"
echo "Log     → $LOG_FILE"
echo "======================================================================"

for pr in "${PRUNING_RATIOS[@]}"; do
  for fw in "${FORGET_WEIGHTS[@]}"; do
    for p1 in "${PHASE1_EPOCHS[@]}"; do
      for p2 in "${PHASE2_EPOCHS[@]}"; do
        for re in "${REFINEMENT_EPOCHS[@]}"; do
          for gm in "${USE_GRADIENT_MASKING[@]}"; do
            ((RUN_COUNT++))

            if is_already_tested "$pr" "$fw" "$p1" "$p2" "$re" "$gm"; then
                echo "[$RUN_COUNT/$TOTAL] Skipping (already done)"
                continue
            fi

            echo "[$RUN_COUNT/$TOTAL] RUNNING → pr=$pr fw=$fw p1=$p1 p2=$p2 ref=$re mask=$gm"
            echo "[$RUN_COUNT/$TOTAL] $(date)" | tee -a "$LOG_FILE"

            TEMP=$(mktemp --suffix=.json)
            EXP_ID=$(create_temp_config "$TEMP" "$pr" "$fw" "$p1" "$p2" "$re" "$gm" "$RUN_COUNT")

            echo "   Experiment ID: $EXP_ID" | tee -a "$LOG_FILE"

            set +e
            if python3 experiments/run_experiments.py --config "$TEMP" --verbose > >(tee -a "$LOG_FILE") 2> >(tee -a "$LOG_FILE" >&2); then
                METRICS=$(extract_metrics "$EXP_ID")
                if [[ $METRICS == ERROR ]]; then
                    echo "$RUN_COUNT,$pr,$fw,$p1,$p2,$re,$gm,ERROR,ERROR,ERROR,ERROR,ERROR,ERROR,ERROR,ERROR,$EXP_ID" >> "$RESULTS_CSV"
                else
                    echo "$RUN_COUNT,$pr,$fw,$p1,$p2,$re,$gm,$METRICS,$EXP_ID" >> "$RESULTS_CSV"
                    echo "   Success – metrics saved" | tee -a "$LOG_FILE"
                fi
            else
                echo "$RUN_COUNT,$pr,$fw,$p1,$p2,$re,$gm,FAILED,FAILED,FAILED,FAILED,FAILED,FAILED,FAILED,FAILED,$EXP_ID" >> "$RESULTS_CSV"
                echo "   Failed" | tee -a "$LOG_FILE"
            fi
            set -e

            rm -f "$TEMP"
            sync
            echo "" | tee -a "$LOG_FILE"
          done
        done
      done
    done
  done
done

# ============================= Final Ranking =================================
echo
echo "======================================================================"
echo "SEARCH FINISHED – Top 10 configurations (closest to ideal point)"
echo "======================================================================"

python3 - <<PY "${RESULTS_CSV}"
import csv, math
rows = []
with open("$1") as f:
    for r in csv.DictReader(f):
        if r['unlearning_effectiveness'] not in ['ERROR','FAILED','N/A','']:
            r['dist'] = float(r['pareto_distance_to_ideal'])
            rows.append(r)

rows.sort(key=lambda x: x['dist'])

print(f"{'#':<3} {'Prune':<6} {'F-w':<5} {'P1':<3} {'P2':<3} {'Ref':<4} {'Mask':<5} {'Forget↓':<8} {'Retain↑':<8} {'Eff↑':<7} {'Dist↓':<8} ID")
print("-" * 90)
for i, r in enumerate(rows[:10], 1):
    print(f"{i:<3} {float(r['pruning_ratio'])*100:4.1f}% {r['forget_weight']:<5} {r['phase1_epochs']:<3} {r['phase2_epochs']:<3} {r['refinement_epochs']:<4} {r['use_gradient_masking']:<5} "
          f"{float(r['forget_accuracy'])*100:5.1f}% {float(r['retain_accuracy'])*100:5.1f}% {float(r['unlearning_effectiveness'])*100:4.1f}% {r['pareto_distance_to_ideal']:<8} {r['experiment_id']}")
PY

echo
echo "Results CSV: $RESULTS_CSV"
echo "Log file   : $LOG_FILE"
echo "Just re-run this script anytime to continue where it left off!"
echo "======================================================================"