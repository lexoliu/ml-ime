# ime-lm

The character LM as a stateful [`Transition`] over ONNX Runtime: `charlm.onnx`
advances a batch of beams by one character, `prefill.onnx` reads the context
once per record. Export directories are written by `mlime export char-lm`
(`python/scripts/charlm_sized.py` makes a randomly initialised one at the real
model's size, which is what `examples/pressure.rs` measures with).

## Building the cuda-harness tarball

`cuda-harness.tar.gz` is the T4 leg's payload: `ime-cli` and the `pressure`
example for `x86_64-unknown-linux-gnu` with `--features gpu-cuda`, plus the
ONNX Runtime provider libraries and a bundled glibc toolchain so the binaries
run on the Kaggle image (Ubuntu 22.04, glibc 2.35 — older than the pyke
dist's 2.38 floor). The harness's own README covers the runtime side; this is
the build side.

Build inside an `ubuntu:24.04` container (glibc 2.39, and the ort cache volume
that pyke's dists download into):

```sh
docker run --rm \
    -v "$PWD":/src \
    -v mlime-2404-cargo:/cargo \
    -v mlime-2404-target:/target \
    -v "$HOME/.cache/ort.pyke.io":/root/.cache/ort.pyke.io \
    -e CARGO_HOME=/cargo -e CARGO_TARGET_DIR=/target \
    ubuntu:24.04 bash -euxc '
        apt-get update -qq &&
        apt-get install -y -qq --no-install-recommends \
            build-essential cmake clang curl ca-certificates pkg-config libssl-dev git perl &&
        export PATH=/cargo/bin:$PATH && cd /src &&
        cargo build --release -p ime-lm -p ime-cli \
            --features gpu-cuda --bins --example pressure'
```

Then assemble `out/cuda-harness/`:

```sh
out=out/cuda-harness
mkdir -p "$out/bin" "$out/lib"
cp /target/release/ime-cli /target/release/examples/pressure "$out/bin/"

# Provider libraries, resolved relative to the executable at dlopen time.
cp /root/.cache/ort.pyke.io/dfbin/x86_64-unknown-linux-gnu/*/libonnxruntime_providers_*.so \
    "$out/bin/"

# glibc 2.39 + the GCC 13 runtime from the same image.
cp /lib/x86_64-linux-gnu/{libc.so.6,libm.so.6,libdl.so.2,libpthread.so.0,
    librt.so.1,libutil.so.1,libgcc_s.so.1} \
   /usr/lib/x86_64-linux-gnu/libstdc++.so.6 \
   /lib64/ld-linux-x86-64.so.2 "$out/lib/"
```

`run.sh` execs the bundled `ld-linux` with `lib/` ahead of `LD_LIBRARY_PATH`
(it lives in the tarball; `CUDA_*` libraries still come from the pip-installed
`nvidia-*` wheels on `LD_LIBRARY_PATH`). Finish with
`tar czf cuda-harness.tar.gz -C out cuda-harness`.

A batch-256 lockstep run on the 8-vCPU VM peaks around 10 GB RSS, CPU backend;
the [W, width] rectangular batch keeps the prefix pack at one row per worker,
so nothing scales with `workers × width` on the host.
