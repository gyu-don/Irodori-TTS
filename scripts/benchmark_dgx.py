"""Measure complete synthesis on the installed GPU, including codec and watermark.

Run with: uv run --extra dgx-spark python scripts/benchmark_dgx.py --help
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from irodori_tts.inference_runtime import (  # noqa: E402
    InferenceRuntime,
    RuntimeKey,
    SamplingRequest,
    save_wav,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--precision", choices=["fp32", "bf16"], default="bf16")
    parser.add_argument("--cudnn", choices=["0", "1"], default="0")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--seconds", type=float)
    parser.add_argument("--ref-wav")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    os.environ["IRODORI_SDPA_CUDNN"] = args.cudnn
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    runtime = InferenceRuntime.from_key(
        RuntimeKey(
            checkpoint=str(args.checkpoint.absolute()),
            model_device="cuda",
            codec_device="cuda",
            model_precision=args.precision,
            codec_precision=args.precision,
        )
    )
    torch.cuda.synchronize()
    report = {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(),
        "checkpoint": str(args.checkpoint.absolute()),
        "precision": args.precision,
        "cudnn": args.cudnn,
        "load_seconds": time.perf_counter() - started,
        "watermark_ready": runtime.watermarker.ready,
        "requests": [],
    }
    request = SamplingRequest(
        text="こんにちは。今日はいい天気ですね。音声合成のテストをしています。",
        caption="落ち着いた女性の声で、自然に話す。",
        no_ref=args.ref_wav is None,
        ref_wav=args.ref_wav,
        seconds=args.seconds,
        seed=42,
    )
    for i in range(args.runs):
        started = time.perf_counter()
        result = runtime.synthesize(request, log_fn=print)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        audio = result.audio.float()
        if not torch.isfinite(audio).all() or audio.square().mean() < 1e-8:
            raise RuntimeError("Synthesis produced non-finite or silent audio")
        duration = audio.shape[-1] / result.sample_rate
        output = args.output_dir / f"sample_{i}.wav"
        save_wav(output, audio, result.sample_rate)
        row = {
            "run": i,
            "cold": i == 0,
            "wall_seconds": elapsed,
            "audio_seconds": duration,
            "rtf": elapsed / duration,
            "stage_timings": dict(result.stage_timings),
            "peak": float(audio.abs().max()),
            "rms": float(audio.square().mean().sqrt()),
            "messages": result.messages,
        }
        report["requests"].append(row)
        report["peak_gpu_gib"] = torch.cuda.max_memory_allocated() / 2**30
        (args.output_dir / "results.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n"
        )
        print(json.dumps(row, ensure_ascii=False), flush=True)
    runtime.unload()


if __name__ == "__main__":
    main()
