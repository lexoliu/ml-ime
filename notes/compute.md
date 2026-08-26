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

## Kaggle — WORKING (since 2026-08-26, after account phone verification)
- `machine_shape: NvidiaTeslaT4` in kernel-metadata.json now yields **2x Tesla T4**,
  torch 2.10.0+cu128, CUDA matmul ok. Before verification the same request was
  silently scheduled onto the CPU image — the downgrade is the symptom of an
  unverified account, not a metadata problem.
- 30h/week GPU quota, internet on. Primary training venue; Colab is the burst lane.
