#!/usr/bin/env bash
# Hierarchical Prototype Attention vs. weakly-supervised baselines on
# iNaturalist-2021-mini.
#
# Usage:
#   ./run.sh maskcon     # baseline: MaskCon
#   ./run.sh grafit      # baseline: Grafit
#   ./run.sh bucsfr      # baseline: BuCSFR
#   ./run.sh hpa         # proposed: standalone HPA
#   ./run.sh test        # sanity checks
#   ./run.sh analyze     # evaluation-only hierarchy diagnostics
#   ./run.sh all         # maskcon -> grafit -> bucsfr -> hpa -> analyze
#
# A second argument (or the GPUS env var) selects the GPU(s):
#   ./run.sh hpa 0       # single GPU
#   ./run.sh hpa 0,1     # two GPUs
#   ./run.sh hpa auto    # let Lightning decide (default)
set -euo pipefail

# ---------------------------------------------------------------- environment
export WANDB_API_KEY="${WANDB_API_KEY:-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}"
PYTHON="${PYTHON:-python}"
INAT_TRAIN_METADATA="${INAT_TRAIN_METADATA:-train_mini.json}"
INAT_VAL_METADATA="${INAT_VAL_METADATA:-val.json}"
INAT_TRAIN_DIR="${INAT_TRAIN_DIR:-inat2021/train_mini}"
INAT_VAL_DIR="${INAT_VAL_DIR:-inat2021/val}"
SUPERCLASS="${SUPERCLASS:-}"          # e.g. Insects, to keep the run tractable
HIERARCHY_DIR="${HIERARCHY_DIR:-results/hierarchy}"
# Agglomerative clustering is O(n^2); cap the samples clustered per coarse class.
HIERARCHY_MAX_PER_CLASS="${HIERARCHY_MAX_PER_CLASS:-10000}"
GPUS="${2:-${GPUS:-auto}}"            # "auto", or comma-separated GPU indices
ACCELERATOR="${ACCELERATOR:-auto}"

TRAIN_CAT="${TRAIN_CAT:-order}"       # coarse label the model is trained on
FINE_CAT="${FINE_CAT:-species}"       # hidden label, evaluation only
TEST_CATS=(family genus species)

# Shared across every arm so the only difference is the objective.
COMMON=(
  --model backbone
  --backbone vit_small_patch16_224
  --dataset inat
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
  --accelerator "$ACCELERATOR"
  --devices "$GPUS"
)
if [[ -n "$SUPERCLASS" ]]; then
  COMMON+=(--superclass "$SUPERCLASS")
fi

# ------------------------------------------------------------------ run modes
run_maskcon() {
  echo ">>> Baseline: MaskCon"
  "$PYTHON" train.py "${COMMON[@]}" \
    --maskcon \
    --maskcon_w 0.5 \
    --maskcon_soft_tau 0.1 \
    --maskcon_queue_size 4096 \
    --grafit_views 2
}

run_grafit() {
  echo ">>> Baseline: Grafit"
  "$PYTHON" train.py "${COMMON[@]}" \
    --grafit \
    --grafit_lam 0.5 \
    --grafit_bank \
    --grafit_views 2
}

run_bucsfr() {
  echo ">>> Baseline: BuCSFR"
  "$PYTHON" train.py "${COMMON[@]}" \
    --bucsfr \
    --bucsfr_alpha 0.5 \
    --bucsfr_clusters_per_class 20 \
    --bucsfr_warmup_epochs 10 \
    --bucsfr_queue_size 4096 \
    --grafit_views 2
}

run_hpa() {
  echo ">>> Proposed: Hierarchical Prototype Attention"
  mkdir -p "$HIERARCHY_DIR"
  "$PYTHON" train.py "${COMMON[@]}" \
    --use_hpa \
    --hpa_levels 3 \
    --hpa_heads 1 \
    --hpa_gamma 0.0 \
    --hpa_views 2 \
    --hpa_consistency_weight 1.0 \
    --hpa_assign_tau 0.1 \
    --hpa_target_tau 0.04 \
    --hpa_sinkhorn_iters 3 \
    --hierarchy_update_interval 5 \
    --hierarchy_metric cosine \
    --hierarchy_linkage average \
    --hierarchy_max_samples_per_class "$HIERARCHY_MAX_PER_CLASS" \
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
  maskcon)  run_maskcon ;;
  grafit)   run_grafit ;;
  bucsfr)   run_bucsfr ;;
  hpa)      run_hpa ;;
  test)     run_test ;;
  analyze)  run_analyze ;;
  all)      run_maskcon; run_grafit; run_bucsfr; run_hpa; run_analyze ;;
  *)        echo "Usage: $0 {maskcon|grafit|bucsfr|hpa|test|analyze|all} [gpus]" >&2; exit 1 ;;
esac
