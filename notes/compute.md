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

## Inference backends (user directive, 2026-08-26)

Explore GPU/ANE — never CPU-only. Production model ships with a measured
backend choice: benchmark CPU vs Metal/MPS vs CoreML/ANE (latency per
keystroke at realistic sequence lengths + resident energy) as part of the
milestone-3 acceptance. macOS IMK runs as one resident server process, so
memory permits 300M–1B int8; the binding constraints are keystroke latency
(~30ms perception budget) and battery. For local batch g2pW annotation, try
ort's CoreML EP after CPU parity is established (CPU stays the parity
reference because fp16 backends flip near-ties).

## Azure student credit — verified dead end for GPU (2026-08-26)

$100 "Azure for Students" credit cannot buy GPU: quota requests for both
NCASv3_T4 and NCADS_A100_v4 return `ResourceNotAvailableForOffer` — the
student offer type is excluded from modern GPU families entirely (checked
eastus/westus2/southcentralus/westeurope, all 0/0; the legacy K80 NC family
holds a 6-vCPU quota but its SKUs are retired). The credit remains usable for
CPU/storage only. Escalation compute beyond Kaggle 30h/week therefore means
Colab paid units or a non-student GPU cloud, decided if/when the kill gate
passes and route B needs it.
