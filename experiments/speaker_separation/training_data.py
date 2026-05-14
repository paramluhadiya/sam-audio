"""Shared utilities for synthetic speaker-separation training data."""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torchaudio


AUDIO_EXTENSIONS = {".flac", ".wav", ".mp3", ".ogg"}
EPS = 1.0e-8
LIBRI_LIGHT_SAMPLE_RATE = 16_000


@dataclass(frozen=True)
class SourceClip:
    path: Path
    speaker_id: str
    sample_rate: int
    num_frames: int

    @property
    def duration(self) -> float:
        return self.num_frames / self.sample_rate


def infer_speaker_id(path: Path) -> str:
    stem_parts = path.stem.split("-")
    if stem_parts and stem_parts[0].isdigit():
        return stem_parts[0]

    for part in reversed(path.parts[:-1]):
        if part.isdigit():
            return part

    return path.parent.name


def scan_sources(
    source_dir: Path,
    min_duration: float,
    max_files: int | None = None,
) -> list[SourceClip]:
    json_clips = scan_librilight_json_sources(source_dir, min_duration, max_files)
    if json_clips:
        return json_clips

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


def scan_librilight_json_sources(
    source_dir: Path,
    min_duration: float,
    max_files: int | None = None,
) -> list[SourceClip]:
    json_paths = sorted(source_dir.rglob("*.json"))
    if max_files is not None:
        json_paths = json_paths[:max_files]

    clips: list[SourceClip] = []
    for json_path in json_paths:
        audio_path = matching_audio_path(json_path)
        if audio_path is None:
            continue
        try:
            metadata = json.loads(json_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            print(f"Skipping unreadable metadata {json_path}: {exc}")
            continue

        duration = metadata_duration(metadata)
        if duration < min_duration:
            continue
        speaker_id = str(metadata.get("speaker") or infer_speaker_id(audio_path))
        clips.append(
            SourceClip(
                path=audio_path,
                speaker_id=speaker_id,
                sample_rate=LIBRI_LIGHT_SAMPLE_RATE,
                num_frames=int(math.ceil(duration * LIBRI_LIGHT_SAMPLE_RATE)),
            )
        )
    return clips


def matching_audio_path(json_path: Path) -> Path | None:
    for suffix in AUDIO_EXTENSIONS:
        candidate = json_path.with_suffix(suffix)
        if candidate.exists():
            return candidate
    return None


def metadata_duration(metadata: dict[str, Any]) -> float:
    voice_activity = metadata.get("voice_activity") or []
    if voice_activity:
        return max(float(end) for _, end in voice_activity)
    book_meta = metadata.get("book_meta") or {}
    if "totaltimesecs" in book_meta:
        return float(book_meta["totaltimesecs"])
    return 0.0


def group_by_speaker(clips: list[SourceClip]) -> dict[str, list[SourceClip]]:
    grouped: dict[str, list[SourceClip]] = {}
    for clip in clips:
        grouped.setdefault(clip.speaker_id, []).append(clip)
    return grouped


def choose_segment_offset(
    clip: SourceClip,
    duration: float,
    rng: random.Random,
) -> int:
    source_frames = int(round(duration * clip.sample_rate))
    max_offset = max(0, clip.num_frames - source_frames)
    return rng.randint(0, max_offset) if max_offset > 0 else 0


def load_segment(
    path: str | Path,
    source_sample_rate: int,
    frame_offset: int,
    duration: float,
    output_sample_rate: int,
) -> torch.Tensor:
    source_frames = int(round(duration * source_sample_rate))
    wav, sample_rate = torchaudio.load(
        str(path),
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
    if rms < EPS:
        return wav
    target_rms = 10.0 ** (target_dbfs / 20.0)
    return wav * (target_rms / rms)


def rms_dbfs(wav: torch.Tensor) -> float:
    rms = wav.square().mean().sqrt()
    return 20.0 * math.log10(float(rms) + EPS)


def peak_dbfs(wav: torch.Tensor) -> float:
    peak = wav.abs().max()
    return 20.0 * math.log10(float(peak) + EPS)


def build_sources_from_record(record: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    wav1 = load_segment(
        record["source1_original_path"],
        int(record["source1_sample_rate"]),
        int(record["source1_frame_offset"]),
        float(record["clip_duration"]),
        int(record["sample_rate"]),
    )
    wav2 = load_segment(
        record["source2_original_path"],
        int(record["source2_sample_rate"]),
        int(record["source2_frame_offset"]),
        float(record["clip_duration"]),
        int(record["sample_rate"]),
    )

    if record.get("rms_dbfs") is not None:
        wav1 = normalize_rms(wav1, float(record["rms_dbfs"]))
        wav2 = normalize_rms(wav2, float(record["rms_dbfs"]))
    return wav1, wav2


def layout_sources(
    wav1: torch.Tensor,
    wav2: torch.Tensor,
    clip_duration: float,
    prompt_duration: float,
    sample_rate: int,
    headroom: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float, float]:
    prompt_frames = int(round(prompt_duration * sample_rate))
    clip_frames = int(round(clip_duration * sample_rate))
    total_frames = clip_frames + prompt_frames

    ref1 = torch.zeros(1, total_frames)
    ref2 = torch.zeros(1, total_frames)
    ref1[:, :clip_frames] = wav1[:, :clip_frames]
    ref2[:, prompt_frames : prompt_frames + clip_frames] = wav2[:, :clip_frames]
    mix = ref1 + ref2

    mix_peak = max(float(mix.abs().max()), EPS)
    mix_gain = min(headroom / mix_peak, 1.0)
    if mix_gain < 1.0:
        ref1 = ref1 * mix_gain
        ref2 = ref2 * mix_gain
        mix = mix * mix_gain
    return mix, ref1, ref2, mix_peak, mix_gain


def build_training_tensors(record: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    wav1, wav2 = build_sources_from_record(record)
    mix, ref1, ref2, _, _ = layout_sources(
        wav1=wav1,
        wav2=wav2,
        clip_duration=float(record["clip_duration"]),
        prompt_duration=float(record["prompt_duration"]),
        sample_rate=int(record["sample_rate"]),
        headroom=float(record["headroom"]),
    )
    if int(record["target_source"]) == 1:
        return mix, ref1, ref2
    return mix, ref2, ref1


def write_audio(path: Path, wav: torch.Tensor, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(path), wav.clamp(min=-1.0, max=1.0).cpu(), sample_rate)
