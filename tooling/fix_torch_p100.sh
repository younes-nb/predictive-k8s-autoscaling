#!/usr/bin/env bash
# Fix PyTorch CUDA build for Tesla P100 (sm_60, CC 6.0) nodes.
#
# Problem: default `pip install -r requirements.txt` pulls torch+cu130 from
# PyPI, which only ships kernels for sm_75+. On a P100 this fails with:
#   torch.AcceleratorError: CUDA error: no kernel image is available
#   for execution on the device
# Fix: reinstall the SAME torch version built against CUDA 12.6, which still
# ships sm_50/sm_60/sm_70 kernels.
#
# No node restart / reboot is needed: this only swaps userspace pip wheels
# (torch + nvidia-*-cu12). The NVIDIA driver (580.x, CUDA 13.0-capable)
# already supports the CUDA 12.6 runtime bundled in the wheels.
#
# Usage:
#   bash tooling/fix_torch_p100.sh [--venv /path/to/venv] [--torch-version 2.14.0+cu126]
#   # or with the venv already activated:
#   bash tooling/fix_torch_p100.sh
#
# After the fix, verify with:
#   python training/train.py --checkpoint_path /tmp/model.pt --model_type lstm \
#     --preprocess_approach none --hyperparam_optimizer none --loss_mode joint_mse \
#     --last_step_only --epochs 2 --train_pct 5 --val_pct 5 --msname MS_61753
set -euo pipefail

VENV=""
TORCH_VER="2.14.0+cu126"
CU_INDEX="https://download.pytorch.org/whl/cu126"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --venv) VENV="$2"; shift 2 ;;
    --torch-version) TORCH_VER="$2"; shift 2 ;;
    -h|--help) sed -n '1,30p' "$0"; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; exit 1 ;;
  esac
done

if [[ -n "$VENV" ]]; then
  PIP="$VENV/bin/pip"
  PY="$VENV/bin/python"
else
  PIP="pip"
  PY="python3"
  if [[ -n "${VIRTUAL_ENV:-}" && -x "$VIRTUAL_ENV/bin/pip" ]]; then
    PIP="$VIRTUAL_ENV/bin/pip"
    PY="$VIRTUAL_ENV/bin/python"
  fi
fi

echo "=== GPU ==="
nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv 2>/dev/null || nvidia-smi 2>&1 | head -n 15
echo
echo "=== torch before fix ==="
"$PY" -c "import torch; print('torch:', torch.__version__, '| cuda:', torch.version.cuda)" 2>&1 || true

echo
echo "=== Installing torch==${TORCH_VER} from ${CU_INDEX} ==="
# NOTE: use --index-url (not -r requirements.txt --index-url), because the
# PyTorch index only hosts torch/torchvision/torchaudio. Install everything
# else from PyPI first, then pin torch to the cu126 build afterwards so a
# plain `pip install -r requirements.txt` (which would pull +cu130) can't
# silently overwrite it.
"$PIP" install "torch==${TORCH_VER}" --index-url "$CU_INDEX"

echo
echo "=== Verifying (arch list must contain sm_60; LSTM .to(cuda) must pass) ==="
"$PY" -c "
import torch
print('torch:', torch.__version__)
print('cuda runtime:', torch.version.cuda)
print('available:', torch.cuda.is_available())
archs = torch.cuda.get_arch_list()
print('archs:', archs)
assert torch.cuda.is_available(), 'CUDA not available after fix'
assert 'sm_60' in archs, f'cu126 build expected sm_60 in arch list, got {archs}'
print('device:', torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
m = torch.nn.LSTM(input_size=8, hidden_size=32, num_layers=2, batch_first=True).to('cuda')
x = torch.randn(4, 12, 8, device='cuda')
y, _ = m(x)
assert tuple(y.shape) == (4, 12, 32), y.shape
print('LSTM forward on GPU: OK', tuple(y.shape))
print('FIX VERIFIED')
"

echo
echo "Done. No reboot needed (userspace-only change)."
echo "IMPORTANT: always install torch with the command above AFTER"
echo "'pip install -r requirements.txt', because requirements.txt has"
echo "'torch>=2.13.0' which resolves to +cu130 (no sm_60 kernels) on PyPI."
