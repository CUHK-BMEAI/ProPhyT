#!/bin/bash
set -euo pipefail

# Run subject-level 5-fold CV end to end: train folds, test folds, then pooled summary.
# Usage:
#   bash run_train_test_ctp_prompt_5fold.sh all
#   bash run_train_test_ctp_prompt_5fold.sh 0

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FOLD_SPEC="${1:-${FOLD:-all}}"

bash "${SCRIPT_DIR}/run_train_ctp_prompt_5fold.sh" "${FOLD_SPEC}"
bash "${SCRIPT_DIR}/run_test_ctp_prompt_5fold.sh" "${FOLD_SPEC}"
