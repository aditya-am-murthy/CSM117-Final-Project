#!/bin/bash
set -euo pipefail
#!/bin/bash
set -euo pipefail

echo -e "\nFINAL BULLETPROOF HYPERPARAM SEARCH — THIS ONE WORKS 100%\n"

rm -rf ./test_search && mkdir -p ./test_search
CSV="./test_search/results.csv"
LOG="./test_search/log.txt"

cat > "$CSV" <<EOF
run_id,pruning_ratio,forget_weight,status,experiment_id
