#!/usr/bin/env bash
# One-command bring-up for a fresh pod. Set the connection in scripts/pod.env first
# (JEPA_POD_HOST / JEPA_POD_PORT / JEPA_POD_KEY), then:  bash scripts/pod_kickoff.sh
#
# Syncs the repo, builds the env, and runs the verifications + probes that don't
# need a trained model. Training/eval is a separate step (see the end).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
POD=scripts/pod.sh

echo "== 1. sync source =="
for f in pyproject.toml \
         src/jepa_v2/__init__.py src/jepa_v2/config.py src/jepa_v2/programl_compat.py \
         src/jepa_v2/vicreg.py src/jepa_v2/splits.py src/jepa_v2/exebench.py \
         src/jepa_v2/data.py src/jepa_v2/loss.py src/jepa_v2/model.py \
         scripts/setup_pod.sh \
         scripts/probe_exebench.py scripts/build_cache.py scripts/train.py \
         scripts/eval_disentangle.py scripts/probe_headroom.py \
         tests/test_vicreg.py tests/test_loss.py tests/test_data.py \
         tests/test_splits.py tests/test_model.py; do
  bash "$POD" put "$f" "$f" >/dev/null && echo "  put $f"
done

echo "== 2. setup env (programl, PyG, datasets, compilers — idempotent, slow first time) =="
bash "$POD" run 'bash scripts/setup_pod.sh 2>&1 | tail -3'

echo "== 3. verify programl =="
bash "$POD" run 'python3 -c "import jepa_v2.programl_compat, programl as pg; g=pg.to_networkx(pg.from_cpp(\"int f(){return 1;}\")); print(\"PROGRAML_OK nodes\", g.number_of_nodes())" 2>&1 | tail -2'

echo "== 4. unit tests =="
bash "$POD" run 'PYTHONPATH=src python3 -m pytest tests/ -q 2>&1 | tail -3'

echo "== 5. headroom probe (x86, gcc/clang) =="
bash "$POD" run 'python3 scripts/probe_headroom.py --out headroom_x86.json 2>&1 | grep -E "HEADROOM::COMPILER|HEADROOM::KERNEL|HEADROOM::GATE"'

cat <<'EOF'

== done. Next (training — needs the cache, ~30-40 min build) ==
  bash scripts/pod.sh run 'cd /workspace/jepa-v2 && . .venv/bin/activate && nohup bash -c "python3 scripts/build_cache.py --n 8000 --split train_real_compilable --pool all --min-nodes 16 --out data/cache_div && python3 scripts/train.py --cache data/cache_div --epochs 50 --batch-programs 128 --out checkpoints/div && python3 scripts/eval_disentangle.py --ckpt checkpoints/div/encoder.pt --cache data/cache_div --split test --out eval_div" > run.log 2>&1 & echo PID $!'
  # poll: bash scripts/pod.sh run 'tail -4 run.log'
EOF
