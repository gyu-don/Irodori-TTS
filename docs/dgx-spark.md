# DGX Spark update and operation

Verified on 2026-11-14, NVIDIA GB10 (SM 12.1), ARM64 Linux, driver 580.178.04.
Upstream: [`3822056`](https://github.com/Aratako/Irodori-TTS/commit/3822056),
2026-11-09. The previous local base was `2708d3c` (v2).

## Recent upstream changes

- v4/v4.1 unify voice cloning and caption-based VoiceDesign, support multiple
  reference clips, and predict output duration. v4.1-Small improves the duration
  predictor. v2/v3 inference remains supported.
- September: MeanFlow distillation (four steps by default) and v4-Large support.
  v4-Large uses a T5Gemma 2 encoder and approximately 3.29B parameters.
- October 25: cross-request batching, cached reference latents, rare-token
  splitting for Small, and an option to skip cuDNN's per-shape attention planning.
- November 9: an opt-in two-pass length check reduces overlong RF Small outputs.

Sources: [upstream history](https://github.com/Aratako/Irodori-TTS/commits/main/),
[Small](https://huggingface.co/Aratako/Irodori-TTS-v4.1-Small),
[MeanFlow](https://huggingface.co/Aratako/Irodori-TTS-v4.1-Small-MF),
[Large](https://huggingface.co/Aratako/Irodori-TTS-v4-Large).

## Installation and launch

```bash
uv sync --locked --extra dgx-spark

# VoiceDesign UI, default model v4.1-Small; localhost:7861
./scripts/dgx-spark.sh

# Voice cloning UI; localhost:7860
./scripts/dgx-spark.sh gradio_app.py

# Four-step MeanFlow, without reference audio
./scripts/dgx-spark.sh infer.py \
  --hf-checkpoint Aratako/Irodori-TTS-v4.1-Small-MF \
  --text "こんにちは。今日はいい天気ですね。" \
  --caption "落ち着いた女性の声で、自然に話す。" \
  --no-ref --output-wav outputs/meanflow.wav
```

For cloning, replace `--no-ref` with `--ref-wav path/to/reference.wav`.
In either UI, change the checkpoint field to select MeanFlow or Large. Leave
Num Steps blank to use the model default (RF: 40, MeanFlow: 4).

The launcher selects the locked CUDA 13 backend and sets
`IRODORI_SDPA_CUDNN=0` unless explicitly overridden. With BF16, skipping cuDNN
reduced the first Small request from 1.078 s to 1.011 s in the final test, while
warm requests were similar (0.678 s versus 0.683 s). The initial comparison
showed a larger first-request difference (1.717 s versus 1.012 s); kernel and
plan caches affect startup measurements. Skipping per-shape planning favors
interactive requests with varying lengths. To use cuDNN:
`IRODORI_SDPA_CUDNN=1 ./scripts/dgx-spark.sh`.

CLI and both UIs default to BF16 when the selected CUDA device is SM 12.1.
Explicit `--model-precision fp32 --codec-precision fp32` still works; CPU and
other GPU defaults are unchanged. `RuntimeKey` callers continue to choose their
own precision explicitly.

After installation, direct commands can use `uv run --no-sync ...` or
`uv run --extra dgx-spark ...`. A plain `uv sync` does not select the Spark
backend. The `dgx-spark` extra is mutually exclusive with the other backends.

## Compatibility decisions

| Previous local change | Decision |
| --- | --- |
| Replace scripted DACVAE Snake when SM 12.1 is absent from PyTorch's architecture list | Removed. CUDA 13's NVRTC executes the original scripted activation successfully in FP32 and BF16, including full encode/decode inference. The architecture-list test was too broad. |
| Exclude TorchCodec on ARM64 Linux | Retained. The upstream-pinned 0.10 wheel has no Linux ARM64 build. |
| Keep datasets below 4 on ARM64 Linux | Retained for audio dataset decoding without TorchCodec. Other platforms use upstream constraints. |
| Fall back to SoundFile on missing TorchCodec | Retained for loading and saving. Handles `ImportError` as well as `RuntimeError`. |
| Convert audio to CPU FP32 before saving | Upstream now does this; use that implementation, with the missing-codec fallback and mono-vector support. |
| CUDA 12.8 only | Added a separate `dgx-spark` CUDA 13 extra, keeping the other upstream backends available. |
| Full FP32 allocation before BF16 transfer | Avoid unused random initialization, release the checkpoint dictionary after loading, and cast on CPU before GPU transfer. Large previously ran out of memory alongside the resident workload; it now loads and synthesizes successfully. |

Installed runtime: PyTorch/TorchAudio `2.10.0+cu130`, CUDA 13.0,
Transformers `5.12.1`, Hugging Face Hub `1.23.0`, datasets `3.6.0`,
torchao `0.16.0`. Dependencies are recorded in `uv.lock`.

PyTorch 2.10 still emits a capability-range warning for GB10: its architecture
list ends with `sm_120` / `compute_120`, rather than an explicit `sm_121` binary.
The warning is not suppressed. The scripted activation and full inference paths
were exercised on the physical GB10; this is a verified working configuration,
not a claim that all PyTorch operations or training modes are validated.

`uv pip check` reports one vendor metadata issue: the ARM64
`nvidia-cusparselt-cu13==0.8.0` wheel internally declares
`manylinux2014_sbsa` although its filename says `manylinux2014_aarch64`.
Its ELF library is AArch64, loads successfully, and PyTorch reports cuSPARSELt
available. The exact version is required by the official PyTorch wheel. The
vendor metadata has been left intact.

## Installed models

Weights and bundled tokenizers are in the normal Hugging Face cache. Existing
v2 weights and the shared codec were already at the latest revision and remain
available. Small and MeanFlow use MIT; Large uses the Gemma terms included in
its downloaded model directory.

| Repository under `Aratako/` | Revision | Status |
| --- | --- | --- |
| Irodori-TTS-500M-v2 | `8fd631cafb911dde466bc30dd558a0dc55e8ccae` | Existing, current |
| Irodori-TTS-500M-v2-VoiceDesign | `456e55708e7183f5c7faa1448209d54aa8991451` | Existing, current |
| Semantic-DACVAE-Japanese-32dim | `47376ee24834d7a05a48ebabfe3cde29b3c5e214` | Existing, current |
| Irodori-TTS-v4.1-Small | `2b28324dc263ed5e6638b3cf3dd94c82ead07b4b` | Added |
| Irodori-TTS-v4.1-Small-MF | `ccc78f5d480b6e51b69b2d5042a14c4da04fea6e` | Added |
| Irodori-TTS-v4-Large | `2e0c55428ce97268a507f1feeb2478f8d9148e8b` | Added |

SilentCipher and its watermark model are also installed and were enabled during
the measurements.

## Validation and measurements

The reproducible benchmark is `scripts/benchmark_dgx.py`. It records model-load
time separately, then performs three complete requests, including codec decode
and SilentCipher watermarking. Warm results are the mean of requests 2 and 3.
The same Japanese text, caption and seed 42 are used throughout. The reference
test uses a generated sample from the Small run as an encode/cache smoke test.
It does not measure similarity to a human speaker.

| Model / settings | Load | First request | Warm request | Audio duration | Peak PyTorch GPU allocation |
| --- | ---: | ---: | ---: | ---: | ---: |
| Small, FP32, cuDNN | 19.280 s | 2.127 s | 1.653 s | 7.24 s | 4.38 GiB |
| Small, BF16, cuDNN | 4.955 s | 1.078 s | 0.678 s | 7.24 s | 2.48 GiB |
| Small, BF16, cuDNN off | 5.500 s | 1.011 s | 0.683 s | 7.24 s | 2.48 GiB |
| MeanFlow, BF16, cuDNN off, reference | 4.733 s | 0.634 s | 0.189 s | 6.16 s | 2.41 GiB |
| Large, BF16, cuDNN off | 8.144 s | 2.343 s | 1.828 s | 5.72 s | 7.30 GiB |

The final Small BF16 configuration is approximately 2.4 times faster than FP32
for warm requests with the same text and output duration. Raw final reports are
in `small-fp32-final/`, `small-bf16-cudnn-final/`, `small-final/`,
`meanflow-final/`, and `large-bf16-efficient/` under the output directory.

These are measurements on this machine under its current workload, not isolated
hardware benchmarks. Another GPU process was resident during testing. Peak
allocation covers this PyTorch process, not total unified-memory use. Automatic
duration differs between models, so the MeanFlow result is not a controlled
quality or speed comparison against the RF run. Audio was checked for finite
values and non-silence; no listening evaluation was performed.

Raw logs, JSON measurements, generated WAVs, package inventory, and exact cache
paths are saved under `outputs/update-2026-11-14/` (ignored by Git).

Both Gradio UIs were launched on temporary localhost ports, their HTTP config
endpoints returned 200, and both precision selectors defaulted to BF16. The
standard CLI launcher generated a MeanFlow WAV using the normal Hub model name.
Both legacy v2 checkpoints also synthesized and saved WAVs with the updated
code and environment. Four unit tests cover missing-codec fallback, stereo and
mono saving, device-dependent precision defaults, and preservation of complex
RoPE/integer buffers during conversion. A small torchao INT8 model also passed
the precision-conversion smoke check. Ruff and Python compilation passed for
the changed code. datasets 3.6 decoded a WAV through the dataset audio column;
the training and manifest-preparation CLIs also loaded their help successfully.

```bash
uv run --extra dgx-spark --extra dev python -m unittest discover -s tests -v
uv run --extra dgx-spark python scripts/benchmark_dgx.py \
  --checkpoint /path/to/model.safetensors --precision bf16 --cudnn 0 \
  --output-dir outputs/benchmark
```

The pre-update Git state is available at `backup/dgx-spark-before-2026-11-14`
(`bcd8517`). Keep its Python environment paired with that revision when restoring;
Transformers 5 in the new environment targets the updated code.
The original environment is saved at
`outputs/update-2026-11-14/venv-before/`; move it back to `.venv` when restoring
the old code, since environment scripts contain absolute paths. The active
`.venv` is the verified new CUDA 13 environment.
