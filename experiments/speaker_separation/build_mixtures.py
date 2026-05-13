"""Build synthetic two-speaker mixtures from single-speaker speech clips.

Each generated mixture has this layout:

    speaker 1: [0.0, clip_duration]
    speaker 2: [prompt_duration, clip_duration + prompt_duration]

With the default one-second prompt duration, the first second contains only
speaker 1, the last second contains only speaker 2, and the middle region is
the overlap used for the hardest part of the evaluation.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path

import torch
import torchaudio


AUDIO_EXTENSIONS = {".flac", ".wav", ".mp3", ".ogg"}


@dataclass(frozen=True)
class SourceClip:
    path: Path
    speaker_id: str
    sample_rate: int
    num_frames: int

    @property
    def duration(self) -> float:
        return self.num_frames / self.sample_rate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        type=Path,
        required=True,
        help="Directory containing single-speaker Libri-Light/LibriSpeech audio files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where mixtures, references, and manifest.jsonl will be written.",
    )
    parser.add_argument("--num-mixtures", type=int, default=100)
    parser.add_argument("--clip-duration", type=float, default=10.0)
    parser.add_argument("--prompt-duration", type=float, default=1.0)
    parser.add_argument("--sample-rate", type=int, default=48_000)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument(
        "--rms-dbfs",
        type=float,
        default=-23.0,
        help="RMS level applied to each source before mixing. Use --no-rms-normalize to disable.",
    )
    parser.add_argument(
        "--no-rms-normalize",
        action="store_true",
        help="Disable per-source RMS normalization before addition.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Optional cap on scanned audio files for quick dry runs.",
    )
    parser.add_argument(
        "--description",
        default="speech",
        help="Text description stored in the manifest for optional text+span prompting.",
    )
    return parser.parse_args()


def infer_speaker_id(path: Path) -> str:
    stem_parts = path.stem.split("-")
    if stem_parts and stem_parts[0].isdigit():
        return stem_parts[0]

    for part in reversed(path.parts[:-1]):
        if part.isdigit():
            return part

    # Fallback for non-LibriSpeech-style paths. This is deliberately stable,
    # but callers should prefer standard Libri paths where possible.
    return path.parent.name


def scan_sources(source_dir: Path, min_duration: float, max_files: int | None) -> list[SourceClip]:
    candidates = sorted(
        path
        for path in source_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS
    )
    if max_files is not None:
        candidates = candidates[:max_files]

    clips: list[SourceClip] = []
    for path in candidates:
        try:
            info = torchaudio.info(str(path))
        except RuntimeError as exc:
            print(f"Skipping unreadable audio {path}: {exc}")
            continue
        if info.num_frames <= 0 or info.sample_rate <= 0:
            continue
        clip = SourceClip(
            path=path,
            speaker_id=infer_speaker_id(path),
            sample_rate=info.sample_rate,
            num_frames=info.num_frames,
        )
        if clip.duration >= min_duration:
            clips.append(clip)

    return clips


def group_by_speaker(clips: list[SourceClip]) -> dict[str, list[SourceClip]]:
    grouped: dict[str, list[SourceClip]] = {}
    for clip in clips:
        grouped.setdefault(clip.speaker_id, []).append(clip)
    return grouped


def load_random_segment(
    clip: SourceClip,
    duration: float,
    output_sample_rate: int,
    rng: random.Random,
) -> torch.Tensor:
    source_frames = int(round(duration * clip.sample_rate))
    max_offset = max(0, clip.num_frames - source_frames)
    frame_offset = rng.randint(0, max_offset) if max_offset > 0 else 0

    wav, sample_rate = torchaudio.load(
        str(clip.path),
        frame_offset=frame_offset,
        num_frames=source_frames,
    )
    wav = wav.mean(dim=0, keepdim=True)
    if sample_rate != output_sample_rate:
        wav = torchaudio.functional.resample(wav, sample_rate, output_sample_rate)

    target_frames = int(round(duration * output_sample_rate))
    if wav.size(-1) < target_frames:
        wav = torch.nn.functional.pad(wav, (0, target_frames - wav.size(-1)))
    return wav[:, :target_frames].float()


def normalize_rms(wav: torch.Tensor, target_dbfs: float) -> torch.Tensor:
    rms = wav.square().mean().sqrt()
    if rms < 1.0e-8:
        return wav
    target_rms = 10.0 ** (target_dbfs / 20.0)
    return wav * (target_rms / rms)


def write_audio(path: Path, wav: torch.Tensor, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(path), wav.clamp(min=-1.0, max=1.0).cpu(), sample_rate)


def build_one_mixture(
    mixture_id: str,
    clip1: SourceClip,
    clip2: SourceClip,
    args: argparse.Namespace,
    rng: random.Random,
) -> dict:
    wav1 = load_random_segment(clip1, args.clip_duration, args.sample_rate, rng)
    wav2 = load_random_segment(clip2, args.clip_duration, args.sample_rate, rng)

    if not args.no_rms_normalize:
        wav1 = normalize_rms(wav1, args.rms_dbfs)
        wav2 = normalize_rms(wav2, args.rms_dbfs)

    prompt_frames = int(round(args.prompt_duration * args.sample_rate))
    clip_frames = int(round(args.clip_duration * args.sample_rate))
    total_frames = clip_frames + prompt_frames

    ref1 = torch.zeros(1, total_frames)
    ref2 = torch.zeros(1, total_frames)
    ref1[:, :clip_frames] = wav1[:, :clip_frames]
    ref2[:, prompt_frames : prompt_frames + clip_frames] = wav2[:, :clip_frames]
    mix = ref1 + ref2

    peak = max(float(mix.abs().max()), 1.0e-8)
    mix_gain = min(0.99 / peak, 1.0)
    if mix_gain < 1.0:
        mix = mix * mix_gain
        ref1 = ref1 * mix_gain
        ref2 = ref2 * mix_gain

    mixture_dir = args.output_dir / mixture_id
    mix_path = mixture_dir / "mix.wav"
    source1_path = mixture_dir / "source1.wav"
    source2_path = mixture_dir / "source2.wav"

    write_audio(mix_path, mix, args.sample_rate)
    write_audio(source1_path, ref1, args.sample_rate)
    write_audio(source2_path, ref2, args.sample_rate)

    return {
        "mixture_id": mixture_id,
        "mix_path": str(mix_path.resolve()),
        "source1_path": str(source1_path.resolve()),
        "source2_path": str(source2_path.resolve()),
        "source1_original_path": str(clip1.path.resolve()),
        "source2_original_path": str(clip2.path.resolve()),
        "speaker1_id": clip1.speaker_id,
        "speaker2_id": clip2.speaker_id,
        "sample_rate": args.sample_rate,
        "clip_duration": args.clip_duration,
        "prompt_duration": args.prompt_duration,
        "mixture_duration": args.clip_duration + args.prompt_duration,
        "overlap_start": args.prompt_duration,
        "overlap_end": args.clip_duration,
        "source1_anchor": ["+", 0.0, args.prompt_duration],
        "source2_anchor": [
            "+",
            args.clip_duration,
            args.clip_duration + args.prompt_duration,
        ],
        "description": args.description,
        "rms_dbfs": None if args.no_rms_normalize else args.rms_dbfs,
        "mix_gain": mix_gain,
    }


def main() -> None:
    args = parse_args()
    if args.prompt_duration <= 0:
        raise ValueError("--prompt-duration must be positive")
    if args.clip_duration <= args.prompt_duration:
        raise ValueError("--clip-duration must be greater than --prompt-duration")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    clips = scan_sources(args.source_dir, args.clip_duration, args.max_files)
    grouped = group_by_speaker(clips)
    speakers = sorted(speaker for speaker, speaker_clips in grouped.items() if speaker_clips)

    if len(speakers) < 2:
        raise RuntimeError(
            f"Need at least two speakers with clips >= {args.clip_duration:.2f}s; found {len(speakers)}."
        )

    manifest_path = args.output_dir / "manifest.jsonl"
    metadata_path = args.output_dir / "metadata.json"
    with manifest_path.open("w") as manifest_file:
        for idx in range(args.num_mixtures):
            speaker1, speaker2 = rng.sample(speakers, 2)
            clip1 = rng.choice(grouped[speaker1])
            clip2 = rng.choice(grouped[speaker2])
            record = build_one_mixture(
                mixture_id=f"mixture_{idx:06d}",
                clip1=clip1,
                clip2=clip2,
                args=args,
                rng=rng,
            )
            manifest_file.write(json.dumps(record) + "\n")

    metadata = {
        "source_dir": str(args.source_dir.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "num_mixtures": args.num_mixtures,
        "num_scanned_clips": len(clips),
        "num_speakers": len(speakers),
        "clip_duration": args.clip_duration,
        "prompt_duration": args.prompt_duration,
        "sample_rate": args.sample_rate,
        "seed": args.seed,
        "rms_dbfs": None if args.no_rms_normalize else args.rms_dbfs,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(
        f"Wrote {args.num_mixtures} mixtures from {len(speakers)} speakers "
        f"to {args.output_dir}."
    )
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
