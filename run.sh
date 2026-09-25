#!/usr/bin/env bash
# Experiments for CE + Hierarchical Prototype Attention on iNaturalist-2021-mini.
#
# Usage:
#   ./run.sh baseline    # CE only
#   ./run.sh hpa         # CE + HPA
#   ./run.sh test        # sanity checks
#   ./run.sh analyze     # evaluation-only hierarchy diagnostics
#   ./run.sh all         # baseline -> hpa -> analyze
set -euo pipefail

# ---------------------------------------------------------------- environment
PYTHON="${PYTHON:-python}"
INAT_TRAIN_METADATA="${INAT_TRAIN_METADATA:-train_mini.json}"
INAT_VAL_METADATA="${INAT_VAL_METADATA:-val.json}"
INAT_TRAIN_DIR="${INAT_TRAIN_DIR:-inat2021/train_mini}"
INAT_VAL_DIR="${INAT_VAL_DIR:-inat2021/val}"
SUPERCLASS="${SUPERCLASS:-}"          # e.g. Insects, to keep the run tractable
HIERARCHY_DIR="${HIERARCHY_DIR:-results/hierarchy}"

TRAIN_CAT="${TRAIN_CAT:-order}"       # coarse label the model is trained on
FINE_CAT="${FINE_CAT:-species}"       # hidden label, evaluation only
TEST_CATS=(family genus species)

# Shared across both arms so the only difference is the HPA flags.
COMMON=(
  --model backbone
  --backbone vit_small_patch16_224
  --dataset inat
  --cross_entropy
  --train_cat "$TRAIN_CAT"
  --test_cat "${TEST_CATS[@]}"
  --inat_train_metadata "$INAT_TRAIN_METADATA"
  --inat_val_metadata "$INAT_VAL_METADATA"
  --inat_train_dir "$INAT_TRAIN_DIR"
  --inat_val_dir "$INAT_VAL_DIR"
  --img_size 224
  --batch_size 128
  --epochs 60
  --seed 42
)
if [[ -n "$SUPERCLASS" ]]; then
  COMMON+=(--superclass "$SUPERCLASS")
fi

# ------------------------------------------------------------------ run modes
run_baseline() {
  echo ">>> Baseline: cross-entropy only"
  "$PYTHON" train.py "${COMMON[@]}"
}

run_hpa() {
  echo ">>> Proposed: cross-entropy + Hierarchical Prototype Attention"
  mkdir -p "$HIERARCHY_DIR"
  "$PYTHON" train.py "${COMMON[@]}" \
    --use_hpa \
    --hpa_levels 3 \
    --hpa_heads 1 \
    --hpa_gamma 0.0 \
    --hierarchy_warmup_epochs 5 \
    --hierarchy_update_interval 5 \
    --hierarchy_metric cosine \
    --hierarchy_linkage average \
    --hierarchy_snapshot_dir "$HIERARCHY_DIR"
}

run_test() {
  echo ">>> Sanity checks"
  "$PYTHON" -m pytest Tests/test_hpa.py -v
}

# Evaluation only: these metrics must never drive training or model selection.
run_analyze() {
  echo ">>> Hierarchy diagnostics against the hidden '$FINE_CAT' labels"
  local args=(
    --snapshots "$HIERARCHY_DIR"/hierarchy_epoch*.npz
    --inat_metadata "$INAT_TRAIN_METADATA"
    --inat_image_dir "$INAT_TRAIN_DIR"
    --train_cat "$TRAIN_CAT"
    --fine_cat "$FINE_CAT"
  )
  if [[ -n "$SUPERCLASS" ]]; then
    args+=(--superclass "$SUPERCLASS")
  fi
  "$PYTHON" analyze_hierarchy.py "${args[@]}"
}

case "${1:-all}" in
  baseline) run_baseline ;;
  hpa)      run_hpa ;;
  test)     run_test ;;
  analyze)  run_analyze ;;
  all)      run_baseline; run_hpa; run_analyze ;;
  *)        echo "Usage: $0 {baseline|hpa|test|analyze|all}" >&2; exit 1 ;;
esac
