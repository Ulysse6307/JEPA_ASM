#!/bin/bash
#SBATCH --job-name=jepa-m15
#SBATCH --partition=defq
#SBATCH --reservation=Vivatech       # the 18 nodes are held by this hackathon
                                     # reservation; without it the job waits behind
                                     # it forever even though GPUs are free
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16            # feeds 8 DataLoader workers — data loading WAS
                                     # the bottleneck (GPU sat at 13% with workers=0)
#SBATCH --gres=gpu:1                 # team quota is 3 GPU total; start with 1
#SBATCH --hint=nomultithread
#SBATCH --time=04:00:00              # short first run; bump once it's working
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
# ---------------------------------------------------------------------------
# JEPA-IR training on Dalia (IDRIS).
#
# Submit from the repo root on Dalia:
#     sbatch scripts/dalia_train.slurm
#
# IMPORTANT — no clang on Dalia:
#   The C->IR->graph half of the pipeline needs clang, which Dalia does NOT
#   provide. So we DON'T compile here. Instead, graphs are pre-built LOCALLY
#   (scripts/build_graphs.py, where clang exists) into a .pt cache, shipped to
#   $WORK, and this job trains from that cache (--graph-cache). No IR toolchain
#   is touched on the GPU node.
#
# Environment:
#   * GPUs are NVIDIA B200; CUDA 12.8 available as a module.
#   * $HOME is only 3 GB -> venv and checkpoints live under $WORK.
#   * $WORK is NOT auto-set; we set it here for the team project group.
#   * GRAPH_CACHE must point to the pre-built .pt (default below).
# ---------------------------------------------------------------------------
set -euo pipefail

# --- paths ----------------------------------------------------------------
export PROJECT_GROUP=vivatech-jepacode
export WORK=${WORK:-/lustre/work/${PROJECT_GROUP}/${USER}}
REPO_DIR=${REPO_DIR:-$WORK/jepa-ir}            # where this repo lives on Dalia
VENV=$WORK/jepa-ir-venv
GRAPH_CACHE=${GRAPH_CACHE:-$WORK/data/graphs_100k.pt}   # pre-built graphs (no clang)
CKPT_DIR=$WORK/runs/mask15_$SLURM_JOB_ID

echo "[slurm] node=$(hostname) job=$SLURM_JOB_ID gpus=$CUDA_VISIBLE_DEVICES"
echo "[slurm] WORK=$WORK REPO=$REPO_DIR"
mkdir -p "$CKPT_DIR"

# --- modules --------------------------------------------------------------
module load cuda12.8/toolkit/12.8.1 2>/dev/null || module load cuda 2>/dev/null || true
nvidia-smi || true

# --- python env (build once, reuse after) ---------------------------------
# Prefer `uv` if available on the node (fast); else fall back to venv+pip.
if [ ! -d "$VENV" ]; then
  echo "[slurm] creating venv at $VENV"
  # NB: llvmlite NOT needed at train time (we load pre-built graphs, no IR step).
  if command -v uv >/dev/null 2>&1; then
    uv venv "$VENV"
    # shellcheck disable=SC1091
    source "$VENV/bin/activate"
    uv pip install --python "$VENV/bin/python" \
        torch --index-url https://download.pytorch.org/whl/cu128
    uv pip install --python "$VENV/bin/python" \
        torch-geometric numpy pyyaml tqdm matplotlib
  else
    python3 -m venv "$VENV"
    # shellcheck disable=SC1091
    source "$VENV/bin/activate"
    pip install --upgrade pip
    pip install torch --index-url https://download.pytorch.org/whl/cu128
    pip install torch-geometric numpy pyyaml tqdm matplotlib
  fi
else
  # shellcheck disable=SC1091
  source "$VENV/bin/activate"
fi

python -c "import torch; print('[slurm] torch', torch.__version__, 'cuda', torch.cuda.is_available())"

# --- data: pre-built graph cache (built locally with clang, shipped here) ---
if [ ! -f "$GRAPH_CACHE" ]; then
  echo "[slurm] ERROR: graph cache not found at $GRAPH_CACHE"
  echo "        Build it locally and copy it over:"
  echo "          # on your laptop (has clang):"
  echo "          python scripts/fetch_anghabench.py --out data/anghabench --sample 10000"
  echo "          python scripts/build_graphs.py --sources data/anghabench/sample \\"
  echo "              --glob '*.c' --out data/graphs.pt"
  echo "          rsync -avz data/graphs.pt dalia:$GRAPH_CACHE"
  exit 1
fi

# --- train (clang-free: loads pre-built graphs) ----------------------------
cd "$REPO_DIR"
# Make the jepa_ir package importable regardless of CWD / install state.
# (srun may run from a different dir, so train.py's sys.path tweak isn't enough.)
export PYTHONPATH="$REPO_DIR/src:${PYTHONPATH:-}"
srun --export=ALL python scripts/train.py \
    --graph-cache "$GRAPH_CACHE" \
    --epochs 50 --batch-size 512 --num-workers 8 --hidden-dim 256 --num-layers 6 --mask-ratio 0.15 \
    --ckpt-dir "$CKPT_DIR" \
    --device cuda --diag-every 5 --mask-edges

echo "[slurm] done. checkpoints + diagnostics in $CKPT_DIR"
echo "[slurm] pull results locally with:"
echo "  rsync -avz dalia:$CKPT_DIR/ ./runs_from_dalia/job_$SLURM_JOB_ID/"