# Intel XPU Installation

## Environment

Use Python 3.12 and the dedicated XPU requirements file. It installs the
validated Intel XPU PyTorch wheels and excludes CUDA-only FlashAttention.

```bash
python3.12 -m venv .venv-xpu
source .venv-xpu/bin/activate
python -m pip install --upgrade pip "setuptools<82" wheel
python -m pip install -r requirements-xpu.txt
```

LeRobot 0.3.3 requires PyTorch earlier than 2.8 and is not included in this
latest-XPU inference/server environment.

## Verify XPU

```bash
python - <<'PY'
import torch

x = torch.randn(2, 3, device="xpu")
y = x @ x.T
torch.xpu.synchronize()
print(torch.__version__)
print(torch.xpu.is_available())
print(torch.xpu.get_device_name())
print(y.shape)
PY
```

## Device Selection

CUDA is preferred when both CUDA and XPU are available. Otherwise, XPU is
selected automatically when available. Select XPU explicitly with:

```bash
export ZERO_WAM_DEVICE=xpu
```

Valid values are `xpu`, `cuda`, `cpu`, and `auto`. The shared configuration's
`device` setting provides the same control.

VAE and text-encoder CPU offload is enabled automatically outside CUDA. Set
`ZERO_WAM_OFFLOAD=0` to disable it or `ZERO_WAM_OFFLOAD=1` to force it.

## Validate Inference

After downloading the Zero-WAM checkpoint, run:

```bash
PYTHONPATH=. ZERO_WAM_DEVICE=xpu python script/benchmark_zero_wam.py \
  --device xpu \
  --warmups 3 \
  --iterations 10 \
  --artifact-dir benchmark_artifacts
```

Run the reduced-step end-to-end server smoke test with synthetic observations:

```bash
PYTHONPATH=. ZERO_WAM_DEVICE=xpu python script/smoke_server.py \
  --device xpu \
  --output-file benchmark_artifacts/server_xpu_actions.npy
```

This test uses one video and one action denoising step. It validates the full
server path but is not a production-quality rollout. On memory-constrained
Intel GPUs, the transformer runs on XPU while the text encoder and VAE use the
automatic CPU-offload path.

To compare matching CUDA and XPU smoke outputs, copy both `.npy` files to the
same host and run:

```bash
python - <<'PY'
import numpy as np

xpu = np.load("benchmark_artifacts/server_xpu_actions.npy")
cuda = np.load("benchmark_artifacts/server_cuda_actions.npy")
diff = np.abs(xpu.astype(np.float64) - cuda.astype(np.float64))

print("MAE:", diff.mean())
print("Max error:", diff.max())
print(
    "Cosine similarity:",
    np.dot(xpu.ravel(), cuda.ravel())
    / (np.linalg.norm(xpu) * np.linalg.norm(cuda)),
)
PY
```