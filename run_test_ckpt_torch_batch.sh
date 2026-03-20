#!/usr/bin/env bash

# 极简顺序批跑脚本：按下面 CASES 列表逐个调用 test_ckpt_torch.py。
# 用法：bash run_test_ckpt_torch_batch.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEST_SCRIPT="${SCRIPT_DIR}/test/scripts/test_ckpt_torch.py"

PYTHON_BIN="python3"
DIM="2"
CUTOFF="1e-3"
DIS="5.0"
DIS_TYPE="random"

# 设为 1 时仅跑 torch 版本。
TORCH_ONLY=0

# 每项格式：L:qmax:lmax
CASES=(
  # "2:1000:100"
  # "3:1000:100"
  # "4:1200:120"
  "5:4000:600"
  "6:8000:1000"
)

if [[ ! -f "$TEST_SCRIPT" ]]; then
  echo "[ERROR] test script not found: $TEST_SCRIPT"
  exit 2
fi

idx=0
for case in "${CASES[@]}"; do
  idx=$((idx + 1))
  IFS=':' read -r L QMAX LMAX <<< "$case"

  cmd=(
    "$PYTHON_BIN" "$TEST_SCRIPT"
    --L "$L"
    --qmax "$QMAX"
    --lmax "$LMAX"
    --dim "$DIM"
    --cutoff "$CUTOFF"
    --dis "$DIS"
    --dis-type "$DIS_TYPE"
  )
  if (( TORCH_ONLY == 1 )); then
    cmd+=(--torch-only)
  fi

  echo
  echo "[$idx/${#CASES[@]}] L=$L, qmax=$QMAX, lmax=$LMAX"
  echo "Command: ${cmd[*]}"
  "${cmd[@]}"
done

echo
echo "All cases finished."
