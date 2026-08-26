# Training compute status (2026-08-26)

## Colab — WORKING
- `colab run --gpu T4 script.py` verified end to end: T4 (15GB), torch 2.11.0+cu128,
  CUDA matmul ok, VM auto-released after run, exit codes propagate.
- One-shot jobs via `colab run`; long sessions via `colab new -s <name> --gpu T4` +
  `colab exec` (kernel state persists across exec calls; `colab stop` releases).
- Supported GPUs: T4, L4, G4, H100, A100 (tier-gated; T4 confirmed on this account).
- CLI defect fixed locally: `google-colab-cli` 0.6.0 leaves `jupyter-kernel-client`
  unpinned and 1.0 renamed `KernelClient` → `JupyterKernelClient`. Reinstalled with
  `uv tool install google-colab-cli --with "jupyter-kernel-client<1" --force`.
  A fresh machine must repeat that pin until upstream pins it.

## Kaggle — GPU gated
- `kaggle kernels push` works (CLI 2.2.4, OAuth). Kernel executes, but GPU requests
  are silently downgraded to CPU: server-side metadata confirms
  `enable_gpu: true, machine_shape: NvidiaTeslaT4` was recorded, run still got the
  CPU image. Known gate: account phone verification unlocks GPU. Not programmatically
  checkable from here.
- CPU kernels remain useful for data preprocessing jobs (30h/week quota, internet on).
