"""Run SAM Audio span-prompted separation on synthetic speaker mixtures."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any

import torch
import torchaudio

from sam_audio import SAMAudio, SAMAudioProcessor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--candidates", type=int, default=1)
    parser.add_argument(
        "--prompt-mode",
        choices=("span", "text-span"),
        default="span",
        help="Use empty text with spans, or the manifest description plus spans.",
    )
    parser.add_argument(
        "--description",
        default=None,
        help="Optional text prompt override. With --prompt-mode span, this still defaults to empty text.",
    )
    parser.add_argument(
        "--save-residuals",
        action="store_true",
        help="Also write SAM Audio residual predictions for each prompt.",
    )
    parser.add_argument(
        "--keep-rankers",
        action="store_true",
        help="Keep ranker configs when loading the checkpoint. By default rankers are disabled for candidates=1.",
    )
    return parser.parse_args()


def read_manifest(path: Path, limit: int | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
                if limit is not None and len(rows) >= limit:
                    break
    return rows


def load_audio(path: str | Path, sample_rate: int) -> torch.Tensor:
    wav, sr = torchaudio.load(str(path))
    wav = wav.mean(dim=0, keepdim=True)
    if sr != sample_rate:
        wav = torchaudio.functional.resample(wav, sr, sample_rate)
    return wav.float()


def trim_pair(estimate: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    n = min(estimate.size(-1), target.size(-1))
    return estimate[..., :n], target[..., :n]


def slice_seconds(wav: torch.Tensor, start: float, end: float, sample_rate: int) -> torch.Tensor:
    start_frame = max(0, int(round(start * sample_rate)))
    end_frame = min(wav.size(-1), int(round(end * sample_rate)))
    return wav[..., start_frame:end_frame]


def si_sdr_db(estimate: torch.Tensor, target: torch.Tensor, eps: float = 1.0e-8) -> float:
    estimate, target = trim_pair(estimate, target)
    estimate = estimate.reshape(-1)
    target = target.reshape(-1)
    estimate = estimate - estimate.mean()
    target = target - target.mean()
    target_energy = torch.sum(target * target) + eps
    projection = torch.sum(estimate * target) * target / target_energy
    noise = estimate - projection
    ratio = (torch.sum(projection * projection) + eps) / (torch.sum(noise * noise) + eps)
    return 10.0 * math.log10(float(ratio))


def snr_db(estimate: torch.Tensor, target: torch.Tensor, eps: float = 1.0e-8) -> float:
    estimate, target = trim_pair(estimate, target)
    error = estimate - target
    ratio = (target.square().sum() + eps) / (error.square().sum() + eps)
    return 10.0 * math.log10(float(ratio))


def metric_block(
    estimate: torch.Tensor,
    target: torch.Tensor,
    mixture: torch.Tensor,
    sample_rate: int,
    overlap_start: float,
    overlap_end: float,
) -> dict[str, float]:
    estimate, target = trim_pair(estimate, target)
    mixture, _ = trim_pair(mixture, target)

    estimate_overlap = slice_seconds(estimate, overlap_start, overlap_end, sample_rate)
    target_overlap = slice_seconds(target, overlap_start, overlap_end, sample_rate)
    mixture_overlap = slice_seconds(mixture, overlap_start, overlap_end, sample_rate)

    full_si_sdr = si_sdr_db(estimate, target)
    overlap_si_sdr = si_sdr_db(estimate_overlap, target_overlap)
    return {
        "si_sdr": full_si_sdr,
        "snr": snr_db(estimate, target),
        "mixture_si_sdr": si_sdr_db(mixture, target),
        "si_sdr_improvement": full_si_sdr - si_sdr_db(mixture, target),
        "overlap_si_sdr": overlap_si_sdr,
        "overlap_snr": snr_db(estimate_overlap, target_overlap),
        "overlap_mixture_si_sdr": si_sdr_db(mixture_overlap, target_overlap),
        "overlap_si_sdr_improvement": overlap_si_sdr
        - si_sdr_db(mixture_overlap, target_overlap),
    }


def flatten_metrics(prefix: str, metrics: dict[str, float]) -> dict[str, float]:
    return {f"{prefix}_{name}": value for name, value in metrics.items()}


def save_audio(path: Path, wav: torch.Tensor, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(path), wav.detach().cpu().clamp(min=-1.0, max=1.0), sample_rate)


def choose_description(record: dict[str, Any], args: argparse.Namespace) -> str:
    if args.description is not None:
        return args.description
    if args.prompt_mode == "text-span":
        return record.get("description", "speech")
    return ""


def run_one_prompt(
    model: SAMAudio,
    processor: SAMAudioProcessor,
    mix_path: str,
    description: str,
    anchor: list[Any],
    device: torch.device,
    candidates: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch = processor(
        audios=[mix_path],
        descriptions=[description],
        anchors=[[tuple(anchor)]],
    ).to(device)
    with torch.inference_mode():
        result = model.separate(batch, reranking_candidates=candidates)
    return result.target[0].detach().cpu(), result.residual[0].detach().cpu()


def summarize(records: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    numeric: dict[str, list[float]] = {}
    for record in records:
        for key, value in record.items():
            if isinstance(value, bool):
                numeric.setdefault(key, []).append(float(value))
            elif isinstance(value, (int, float)):
                numeric.setdefault(key, []).append(float(value))

    summary: dict[str, dict[str, float]] = {}
    for key, values in sorted(numeric.items()):
        if not values:
            continue
        summary[key] = {
            "mean": statistics.fmean(values),
            "median": statistics.median(values),
            "min": min(values),
            "max": max(values),
        }
        if len(values) > 1:
            summary[key]["stdev"] = statistics.stdev(values)
    return summary


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    model_kwargs: dict[str, Any] = {}
    if not args.keep_rankers and args.candidates == 1:
        model_kwargs["text_ranker"] = None
        model_kwargs["visual_ranker"] = None

    model = SAMAudio.from_pretrained(args.checkpoint_path, **model_kwargs)
    model = model.eval().to(device)
    processor = SAMAudioProcessor.from_pretrained(args.checkpoint_path)
    sample_rate = processor.audio_sampling_rate

    rows = read_manifest(args.manifest, args.limit)
    if not rows:
        raise RuntimeError(f"No rows found in {args.manifest}")

    result_rows: list[dict[str, Any]] = []
    results_jsonl = args.output_dir / "results.jsonl"
    with results_jsonl.open("w") as out:
        for index, record in enumerate(rows):
            mixture_id = record["mixture_id"]
            description = choose_description(record, args)
            mixture_output_dir = args.output_dir / mixture_id

            pred1, residual1 = run_one_prompt(
                model=model,
                processor=processor,
                mix_path=record["mix_path"],
                description=description,
                anchor=record["source1_anchor"],
                device=device,
                candidates=args.candidates,
            )
            pred2, residual2 = run_one_prompt(
                model=model,
                processor=processor,
                mix_path=record["mix_path"],
                description=description,
                anchor=record["source2_anchor"],
                device=device,
                candidates=args.candidates,
            )

            ref1 = load_audio(record["source1_path"], sample_rate)
            ref2 = load_audio(record["source2_path"], sample_rate)
            mix = load_audio(record["mix_path"], sample_rate)

            save_audio(mixture_output_dir / "pred_source1.wav", pred1, sample_rate)
            save_audio(mixture_output_dir / "pred_source2.wav", pred2, sample_rate)
            if args.save_residuals:
                save_audio(
                    mixture_output_dir / "residual_from_source1_prompt.wav",
                    residual1,
                    sample_rate,
                )
                save_audio(
                    mixture_output_dir / "residual_from_source2_prompt.wav",
                    residual2,
                    sample_rate,
                )

            overlap_start = float(record["overlap_start"])
            overlap_end = float(record["overlap_end"])
            speaker1_direct = metric_block(
                pred1, ref1, mix, sample_rate, overlap_start, overlap_end
            )
            speaker2_direct = metric_block(
                pred2, ref2, mix, sample_rate, overlap_start, overlap_end
            )
            speaker1_from_residual = metric_block(
                residual2, ref1, mix, sample_rate, overlap_start, overlap_end
            )
            speaker2_from_residual = metric_block(
                residual1, ref2, mix, sample_rate, overlap_start, overlap_end
            )
            direct_swapped_source1 = metric_block(
                pred1, ref2, mix, sample_rate, overlap_start, overlap_end
            )
            direct_swapped_source2 = metric_block(
                pred2, ref1, mix, sample_rate, overlap_start, overlap_end
            )
            direct_overlap = (
                speaker1_direct["overlap_si_sdr"]
                + speaker2_direct["overlap_si_sdr"]
            ) / 2.0
            swapped_overlap = (
                direct_swapped_source1["overlap_si_sdr"]
                + direct_swapped_source2["overlap_si_sdr"]
            ) / 2.0
            residual_overlap = (
                speaker1_from_residual["overlap_si_sdr"]
                + speaker2_from_residual["overlap_si_sdr"]
            ) / 2.0

            result_record: dict[str, Any] = {
                "index": index,
                "mixture_id": mixture_id,
                "speaker1_id": record["speaker1_id"],
                "speaker2_id": record["speaker2_id"],
                "prompt_mode": args.prompt_mode,
                "direct_overlap_si_sdr_mean": direct_overlap,
                "residual_overlap_si_sdr_mean": residual_overlap,
                "swapped_overlap_si_sdr_mean": swapped_overlap,
                "assignment_correct": direct_overlap >= swapped_overlap,
                "residual_minus_direct_overlap_si_sdr": residual_overlap
                - direct_overlap,
            }
            for metric_name in ("si_sdr", "snr", "overlap_si_sdr", "overlap_snr"):
                result_record[f"speaker1_residual_minus_direct_{metric_name}"] = (
                    speaker1_from_residual[metric_name] - speaker1_direct[metric_name]
                )
                result_record[f"speaker2_residual_minus_direct_{metric_name}"] = (
                    speaker2_from_residual[metric_name] - speaker2_direct[metric_name]
                )

            result_record.update(flatten_metrics("speaker1_direct", speaker1_direct))
            result_record.update(flatten_metrics("speaker2_direct", speaker2_direct))
            result_record.update(
                flatten_metrics("speaker1_residual", speaker1_from_residual)
            )
            result_record.update(
                flatten_metrics("speaker2_residual", speaker2_from_residual)
            )
            result_rows.append(result_record)
            out.write(json.dumps(result_record) + "\n")
            out.flush()

            print(
                f"[{index + 1}/{len(rows)}] {mixture_id}: "
                f"direct overlap SI-SDR={direct_overlap:.2f} dB, "
                f"residual overlap SI-SDR={residual_overlap:.2f} dB, "
                f"residual-direct={residual_overlap - direct_overlap:+.2f} dB"
            )

    summary = {
        "manifest": str(args.manifest),
        "checkpoint_path": args.checkpoint_path,
        "num_examples": len(result_rows),
        "sample_rate": sample_rate,
        "prompt_mode": args.prompt_mode,
        "candidates": args.candidates,
        "metrics": summarize(result_rows),
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Wrote per-example results to {results_jsonl}")
    print(f"Wrote summary to {summary_path}")


if __name__ == "__main__":
    main()
