"""Build filtered held-out mixtures for span-separation evaluation."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from training_data import (
    SourceClip,
    build_sources_from_record,
    choose_segment_offset,
    group_by_speaker,
    layout_sources,
    load_segment,
    peak_dbfs,
    rms_dbfs,
    scan_sources,
    write_audio,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-mixtures", type=int, default=64)
    parser.add_argument("--max-attempts", type=int, default=None)
    parser.add_argument("--clip-duration", type=float, default=10.0)
    parser.add_argument("--prompt-duration", type=float, default=1.0)
    parser.add_argument("--sample-rate", type=int, default=48_000)
    parser.add_argument("--seed", type=int, default=113)
    parser.add_argument("--rms-dbfs", type=float, default=-23.0)
    parser.add_argument("--anchor-min-rms-dbfs", type=float, default=-45.0)
    parser.add_argument("--anchor-min-peak-dbfs", type=float, default=-55.0)
    parser.add_argument("--headroom", type=float, default=0.98)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--description", default="")
    return parser.parse_args()


def source_record(prefix: str, clip: SourceClip, frame_offset: int) -> dict[str, Any]:
    return {
        f"{prefix}_original_path": str(clip.path.resolve()),
        f"{prefix}_speaker_id": clip.speaker_id,
        f"{prefix}_sample_rate": clip.sample_rate,
        f"{prefix}_num_frames": clip.num_frames,
        f"{prefix}_frame_offset": frame_offset,
    }


def candidate_record(
    mixture_id: str,
    clip1: SourceClip,
    clip2: SourceClip,
    args: argparse.Namespace,
    rng: random.Random,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "mixture_id": mixture_id,
        "sample_rate": args.sample_rate,
        "clip_duration": args.clip_duration,
        "prompt_duration": args.prompt_duration,
        "mixture_duration": args.clip_duration + args.prompt_duration,
        "overlap_start": args.prompt_duration,
        "overlap_end": args.clip_duration,
        "rms_dbfs": args.rms_dbfs,
        "headroom": args.headroom,
        "description": args.description,
        "source1_anchor": ["+", 0.0, args.prompt_duration],
        "source2_anchor": [
            "+",
            args.clip_duration,
            args.clip_duration + args.prompt_duration,
        ],
    }
    record.update(
        source_record("source1", clip1, choose_segment_offset(clip1, args.clip_duration, rng))
    )
    record.update(
        source_record("source2", clip2, choose_segment_offset(clip2, args.clip_duration, rng))
    )
    record["speaker1_id"] = record["source1_speaker_id"]
    record["speaker2_id"] = record["source2_speaker_id"]
    return record


def load_anchor(record: dict[str, Any], source_index: int, args: argparse.Namespace):
    prefix = f"source{source_index}"
    source_sample_rate = int(record[f"{prefix}_sample_rate"])
    frame_offset = int(record[f"{prefix}_frame_offset"])
    if source_index == 2:
        frame_offset += int(round((args.clip_duration - args.prompt_duration) * source_sample_rate))
    return load_segment(
        record[f"{prefix}_original_path"],
        source_sample_rate,
        frame_offset,
        args.prompt_duration,
        args.sample_rate,
    )


def add_anchor_stats(record: dict[str, Any], args: argparse.Namespace) -> None:
    for source_index in (1, 2):
        anchor = load_anchor(record, source_index, args)
        record[f"source{source_index}_anchor_rms_dbfs"] = rms_dbfs(anchor)
        record[f"source{source_index}_anchor_peak_dbfs"] = peak_dbfs(anchor)


def rejection_reason(record: dict[str, Any], args: argparse.Namespace) -> str | None:
    for source in ("source1", "source2"):
        if record[f"{source}_anchor_rms_dbfs"] < args.anchor_min_rms_dbfs:
            return f"{source}_anchor_rms"
        if record[f"{source}_anchor_peak_dbfs"] < args.anchor_min_peak_dbfs:
            return f"{source}_anchor_peak"
    return None


def write_mixture(record: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    wav1, wav2 = build_sources_from_record(record)
    mix, ref1, ref2, mix_peak, mix_gain = layout_sources(
        wav1=wav1,
        wav2=wav2,
        clip_duration=float(record["clip_duration"]),
        prompt_duration=float(record["prompt_duration"]),
        sample_rate=int(record["sample_rate"]),
        headroom=float(record["headroom"]),
    )
    mixture_dir = output_dir / record["mixture_id"]
    mix_path = mixture_dir / "mix.wav"
    source1_path = mixture_dir / "source1.wav"
    source2_path = mixture_dir / "source2.wav"
    write_audio(mix_path, mix, int(record["sample_rate"]))
    write_audio(source1_path, ref1, int(record["sample_rate"]))
    write_audio(source2_path, ref2, int(record["sample_rate"]))
    record["mix_path"] = str(mix_path.resolve())
    record["source1_path"] = str(source1_path.resolve())
    record["source2_path"] = str(source2_path.resolve())
    record["mix_peak_pre_gain"] = mix_peak
    record["mix_gain"] = mix_gain
    return record


def main() -> None:
    args = parse_args()
    if args.max_attempts is None:
        args.max_attempts = args.num_mixtures * 30
    if args.prompt_duration <= 0:
        raise ValueError("--prompt-duration must be positive")
    if args.clip_duration <= args.prompt_duration:
        raise ValueError("--clip-duration must be greater than --prompt-duration")
    if not (0.0 < args.headroom <= 1.0):
        raise ValueError("--headroom must be in (0, 1]")

    rng = random.Random(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "manifest.jsonl"
    metadata_path = args.output_dir / "metadata.json"

    clips = scan_sources(args.source_dir, args.clip_duration, args.max_files)
    grouped = group_by_speaker(clips)
    speakers = sorted(speaker for speaker, speaker_clips in grouped.items() if speaker_clips)
    if len(speakers) < 2:
        raise RuntimeError(f"Need at least two speakers; found {len(speakers)}")

    accepted = 0
    attempts = 0
    rejections: dict[str, int] = {}
    with manifest_path.open("w") as handle:
        while accepted < args.num_mixtures and attempts < args.max_attempts:
            attempts += 1
            speaker1, speaker2 = rng.sample(speakers, 2)
            clip1 = rng.choice(grouped[speaker1])
            clip2 = rng.choice(grouped[speaker2])
            record = candidate_record(
                mixture_id=f"mixture_{accepted:06d}",
                clip1=clip1,
                clip2=clip2,
                args=args,
                rng=rng,
            )
            try:
                add_anchor_stats(record, args)
            except RuntimeError as exc:
                rejections["load_error"] = rejections.get("load_error", 0) + 1
                print(f"Rejecting unreadable candidate: {exc}")
                continue
            reason = rejection_reason(record, args)
            if reason is not None:
                rejections[reason] = rejections.get(reason, 0) + 1
                continue

            record = write_mixture(record, args.output_dir)
            handle.write(json.dumps(record) + "\n")
            accepted += 1
            if accepted % 25 == 0:
                print(
                    f"Accepted {accepted}/{args.num_mixtures} after {attempts} attempts"
                )

    metadata = {
        "source_dir": str(args.source_dir.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "manifest_path": str(manifest_path.resolve()),
        "num_mixtures_requested": args.num_mixtures,
        "num_mixtures": accepted,
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
        "headroom": args.headroom,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Wrote {accepted} filtered mixtures to {manifest_path}")
    print(f"Attempts: {attempts}; rejections: {rejections}")


if __name__ == "__main__":
    main()
