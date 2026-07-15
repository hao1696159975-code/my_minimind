#!/usr/bin/env bash
# Phase 0 environment + upstream sanity checks.
# Run from my_minimind repo root on the training server.
set -euo pipefail

echo "== Phase 0 checklist =="
echo "date: $(date -Iseconds 2>/dev/null || date)"
echo "host: $(hostname)"
echo "pwd:  $(pwd)"

echo
echo "-- git (this repo) --"
git rev-parse --short HEAD 2>/dev/null || echo "not a git repo?"
git status -sb || true

echo
echo "-- upstream --"
if [[ -d upstream/.git || -f upstream/.git ]]; then
  git -C upstream rev-parse --short HEAD
  git -C upstream log -1 --oneline
else
  echo "MISSING upstream/. Run:"
  echo "  git clone --depth 1 https://github.com/jingyaogong/minimind.git upstream"
  exit 1
fi

echo
echo "-- python / torch --"
python - <<'PY'
import sys
print("python", sys.version)
try:
    import torch
    print("torch", torch.__version__)
    print("cuda_available", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("device", torch.cuda.get_device_name(0))
        print("vram_gb", round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2))
except Exception as e:
    print("torch import failed:", e)
    sys.exit(1)
PY

echo
echo "-- key upstream files --"
for f in \
  upstream/README.md \
  upstream/model/model_minimind.py \
  upstream/model/model_lora.py \
  upstream/trainer/train_pretrain.py \
  upstream/trainer/train_full_sft.py \
  upstream/eval_llm.py
 do
  if [[ -f "$f" ]]; then echo "OK  $f"; else echo "MISS $f"; fi
done

echo
echo "-- param count (Dense default) --"
python scripts/lab/count_params.py --upstream upstream || true

echo
echo "Phase 0 checklist finished. Fill learning/00_project_map.md §实测记录"
