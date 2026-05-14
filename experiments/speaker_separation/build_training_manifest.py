"""Build a filtered manifest for span-conditioned speaker-separation training."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from training_data import (
    SourceClip,
    build_sources_from_record,
    build_training_tensors,
    choose_segment_offset,
    group_by_speaker,
    layout_sources,
    peak_dbfs,
    rms_dbfs,
    scan_sources,
    write_audio,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-examples", type=int, default=50_000)
    parser.add_argument("--max-attempts", type=int, default=None)
    parser.add_argument("--clip-duration", type=float, default=10.0)
    parser.add_argument("--prompt-duration", type=float, default=1.0)
    parser.add_argument("--sample-rate", type=int, default=48_000)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--rms-dbfs", type=float, default=-23.0)
    parser.add_argument("--anchor-min-rms-dbfs", type=float, default=-35.0)
    parser.add_argument("--anchor-min-peak-dbfs", type=float, default=-25.0)
    parser.add_argument("--max-overlap-rms-delta-db", type=float, default=8.0)
    parser.add_argument("--headroom", type=float, default=0.98)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--description", default="")
    parser.add_argument(
        "--preview-examples",
        type=int,
        default=8,
        help="Write this many audio previews next to the manifest.",
    )
    return parser.parse_args()


def source_record(
    prefix: str,
    clip: SourceClip,
    frame_offset: int,
) -> dict[str, Any]:
    return {
        f"{prefix}_original_path": str(clip.path.resolve()),
        f"{prefix}_speaker_id": clip.speaker_id,
        f"{prefix}_sample_rate": clip.sample_rate,
        f"{prefix}_num_frames": clip.num_frames,
        f"{prefix}_frame_offset": frame_offset,
    }


def candidate_record(
    example_id: str,
    clip1: SourceClip,
    clip2: SourceClip,
    args: argparse.Namespace,
    rng: random.Random,
) -> dict[str, Any]:
    target_source = 1 if rng.random() < 0.5 else 2
    record: dict[str, Any] = {
        "example_id": example_id,
        "sample_rate": args.sample_rate,
        "clip_duration": args.clip_duration,
        "prompt_duration": args.prompt_duration,
        "mixture_duration": args.clip_duration + args.prompt_duration,
        "overlap_start": args.prompt_duration,
        "overlap_end": args.clip_duration,
        "rms_dbfs": args.rms_dbfs,
        "headroom": args.headroom,
        "target_source": target_source,
        "description": args.description,
        "source1_anchor": ["+", 0.0, args.prompt_duration],
        "source2_anchor": [
            "+",
            args.clip_duration,
            args.clip_duration + args.prompt_duration,
        ],
    }
    record.update(source_record("source1", clip1, choose_segment_offset(clip1, args.clip_duration, rng)))
    record.update(source_record("source2", clip2, choose_segment_offset(clip2, args.clip_duration, rng)))
    record["anchor"] = record["source1_anchor"] if target_source == 1 else record["source2_anchor"]
    record["target_speaker_id"] = (
        record["source1_speaker_id"] if target_source == 1 else record["source2_speaker_id"]
    )
    record["residual_speaker_id"] = (
        record["source2_speaker_id"] if target_source == 1 else record["source1_speaker_id"]
    )
    return record


def add_filter_stats(record: dict[str, Any], args: argparse.Namespace) -> dict[str, float]:
    wav1, wav2 = build_sources_from_record(record)
    prompt_frames = int(round(args.prompt_duration * args.sample_rate))
    clip_frames = int(round(args.clip_duration * args.sample_rate))

    source1_anchor = wav1[:, :prompt_frames]
    source2_anchor = wav2[:, clip_frames - prompt_frames : clip_frames]
    source1_overlap = wav1[:, prompt_frames:clip_frames]
    source2_overlap = wav2[:, : clip_frames - prompt_frames]
    mix, _, _, mix_peak, mix_gain = layout_sources(
        wav1=wav1,
        wav2=wav2,
        clip_duration=args.clip_duration,
        prompt_duration=args.prompt_duration,
        sample_rate=args.sample_rate,
        headroom=args.headroom,
    )
    del mix

    return {
        "source1_anchor_rms_dbfs": rms_dbfs(source1_anchor),
        "source1_anchor_peak_dbfs": peak_dbfs(source1_anchor),
        "source2_anchor_rms_dbfs": rms_dbfs(source2_anchor),
        "source2_anchor_peak_dbfs": peak_dbfs(source2_anchor),
        "source1_overlap_rms_dbfs": rms_dbfs(source1_overlap),
        "source2_overlap_rms_dbfs": rms_dbfs(source2_overlap),
        "overlap_rms_delta_db": abs(rms_dbfs(source1_overlap) - rms_dbfs(source2_overlap)),
        "mix_peak_pre_gain": mix_peak,
        "mix_gain": mix_gain,
    }


def rejection_reason(record: dict[str, Any], args: argparse.Namespace) -> str | None:
    for source in ("source1", "source2"):
        if record[f"{source}_anchor_rms_dbfs"] < args.anchor_min_rms_dbfs:
            return f"{source}_anchor_rms"
        if record[f"{source}_anchor_peak_dbfs"] < args.anchor_min_peak_dbfs:
            return f"{source}_anchor_peak"
    if record["overlap_rms_delta_db"] > args.max_overlap_rms_delta_db:
        return "overlap_balance"
    return None


def write_preview(record: dict[str, Any], preview_dir: Path) -> None:
    mix, target, residual = build_training_tensors(record)
    example_dir = preview_dir / record["example_id"]
    write_audio(example_dir / "mix.wav", mix, int(record["sample_rate"]))
    write_audio(example_dir / "target.wav", target, int(record["sample_rate"]))
    write_audio(example_dir / "residual.wav", residual, int(record["sample_rate"]))


def main() -> None:
    args = parse_args()
    if args.max_attempts is None:
        args.max_attempts = args.num_examples * 30
    if args.prompt_duration <= 0:
        raise ValueError("--prompt-duration must be positive")
    if args.clip_duration <= args.prompt_duration:
        raise ValueError("--clip-duration must be greater than --prompt-duration")
    if not (0.0 < args.headroom <= 1.0):
        raise ValueError("--headroom must be in (0, 1]")

    rng = random.Random(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "train_manifest.jsonl"
    metadata_path = args.output_dir / "metadata.json"
    preview_dir = args.output_dir / "previews"

    clips = scan_sources(args.source_dir, args.clip_duration, args.max_files)
    grouped = group_by_speaker(clips)
    speakers = sorted(speaker for speaker, speaker_clips in grouped.items() if speaker_clips)
    if len(speakers) < 2:
        raise RuntimeError(f"Need at least two speakers; found {len(speakers)}")

    accepted = 0
    attempts = 0
    rejections: dict[str, int] = {}
    with manifest_path.open("w") as handle:
        while accepted < args.num_examples and attempts < args.max_attempts:
            attempts += 1
            speaker1, speaker2 = rng.sample(speakers, 2)
            clip1 = rng.choice(grouped[speaker1])
            clip2 = rng.choice(grouped[speaker2])
            record = candidate_record(
                f"train_{accepted:08d}",
                clip1=clip1,
                clip2=clip2,
                args=args,
                rng=rng,
            )
            try:
                record.update(add_filter_stats(record, args))
            except RuntimeError as exc:
                reason = "load_error"
                rejections[reason] = rejections.get(reason, 0) + 1
                print(f"Rejecting unreadable candidate: {exc}")
                continue

            reason = rejection_reason(record, args)
            if reason is not None:
                rejections[reason] = rejections.get(reason, 0) + 1
                continue

            handle.write(json.dumps(record) + "\n")
            accepted += 1
            if accepted <= args.preview_examples:
                write_preview(record, preview_dir)
            if accepted % 1000 == 0:
                print(f"Accepted {accepted}/{args.num_examples} after {attempts} attempts")

    metadata = {
        "source_dir": str(args.source_dir.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "manifest_path": str(manifest_path.resolve()),
        "num_examples_requested": args.num_examples,
        "num_examples": accepted,
        "attempts": attempts,
        "rejections": rejections,
        "num_scanned_clips": len(clips),
        "num_speakers": len(speakers),
        "clip_duration": args.clip_duration,
        "prompt_duration": args.prompt_duration,
        "sample_rate": args.sample_rate,
        "seed": args.seed,
        "rms_dbfs": args.rms_dbfs,
        "anchor_min_rms_dbfs": args.anchor_min_rms_dbfs,
        "anchor_min_peak_dbfs": args.anchor_min_peak_dbfs,
        "max_overlap_rms_delta_db": args.max_overlap_rms_delta_db,
        "headroom": args.headroom,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Wrote {accepted} accepted examples to {manifest_path}")
    print(f"Attempts: {attempts}; rejections: {rejections}")


if __name__ == "__main__":
    main()
