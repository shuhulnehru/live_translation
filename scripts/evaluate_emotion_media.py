from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from pathlib import Path

from app.services.emotion_media import run_emotion_media_pipeline


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    idx = int(round((len(values) - 1) * q))
    return sorted(values)[idx]


async def _evaluate(dataset_path: Path, output_dir: Path, target_language: str) -> dict:
    payload = json.loads(dataset_path.read_text(encoding="utf-8"))
    samples: list[dict] = payload.get("samples", [])
    if not samples:
        raise ValueError("Dataset must contain 'samples' with at least one row")

    latencies: list[float] = []
    fallback_count = 0
    quality_pass_count = 0
    correct_count = 0
    results: list[dict] = []

    for sample in samples:
        prompt = sample.get("prompt_text", "")
        expected = sample.get("expected_emotion")
        start = time.perf_counter()
        result = await run_emotion_media_pipeline(
            prompt_text=prompt,
            output_dir=str(output_dir),
            target_language=target_language,
            min_confidence=0.6,
        )
        elapsed = time.perf_counter() - start
        latencies.append(elapsed)

        detected = (result.get("output", {}) or {}).get("emotion")
        fallback = bool(result.get("fallback_used", False))
        quality_passed = bool((result.get("output", {}) or {}).get("quality_passed", False))
        is_correct = expected is not None and detected == expected
        correct_count += int(is_correct)
        fallback_count += int(fallback)
        quality_pass_count += int(quality_passed)

        results.append(
            {
                "prompt_text": prompt,
                "expected_emotion": expected,
                "detected_emotion": detected,
                "latency_seconds": round(elapsed, 4),
                "quality_passed": quality_passed,
                "fallback_used": fallback,
                "quality_notes": result.get("quality_notes", []),
            }
        )

    total = len(samples)
    metrics = {
        "count": total,
        "latency_seconds": {
            "p50": round(_percentile(latencies, 0.50), 4),
            "p95": round(_percentile(latencies, 0.95), 4),
            "mean": round(statistics.mean(latencies), 4),
        },
        "fallback_rate": round(fallback_count / total, 4),
        "quality_pass_rate": round(quality_pass_count / total, 4),
        "emotion_accuracy": round(correct_count / total, 4) if any(s.get("expected_emotion") for s in samples) else None,
    }
    return {"metrics": metrics, "results": results}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate emotion-aware image+audio pipeline")
    parser.add_argument("--dataset", required=True, help="JSON dataset path with samples")
    parser.add_argument("--output-dir", default="outputs/emotion_media_eval", help="Output directory")
    parser.add_argument("--target-language", default="en", help="TTS language")
    parser.add_argument("--report", default="outputs/emotion_media_eval/report.json", help="Report output path")
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    output_dir = Path(args.output_dir)
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    report = asyncio.run(_evaluate(dataset_path=dataset_path, output_dir=output_dir, target_language=args.target_language))
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved evaluation report to {report_path}")


if __name__ == "__main__":
    main()
