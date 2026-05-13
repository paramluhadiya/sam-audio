"""Generate plots and feature-space diagnostics for span separation results."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torchaudio


EVAL_PAIRS = (
    ("source1_direct", "source1_ref.wav", "pred_source1.wav", "direct"),
    ("source1_residual", "source1_ref.wav", "source1_from_residual.wav", "residual"),
    ("source2_direct", "source2_ref.wav", "pred_source2.wav", "direct"),
    ("source2_residual", "source2_ref.wav", "source2_from_residual.wav", "residual"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-dir", type=Path, required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Optional manifest with overlap_start/overlap_end per mixture.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--sample-rate", type=int, default=48_000)
    parser.add_argument("--n-mels", type=int, default=80)
    parser.add_argument("--n-fft", type=int, default=2048)
    parser.add_argument("--hop-length", type=int, default=512)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_manifest(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None or not path.exists():
        return {}
    return {row["mixture_id"]: row for row in read_jsonl(path)}


def load_audio(path: Path, sample_rate: int) -> torch.Tensor:
    wav, sr = torchaudio.load(str(path))
    wav = wav.mean(dim=0, keepdim=True).float()
    if sr != sample_rate:
        wav = torchaudio.functional.resample(wav, sr, sample_rate)
    return wav


def trim_pair(reference: torch.Tensor, estimate: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    n = min(reference.size(-1), estimate.size(-1))
    return reference[..., :n], estimate[..., :n]


def slice_seconds(wav: torch.Tensor, start: float, end: float, sample_rate: int) -> torch.Tensor:
    start_frame = max(0, int(round(start * sample_rate)))
    end_frame = min(wav.size(-1), int(round(end * sample_rate)))
    return wav[..., start_frame:end_frame]


def mel_distribution(
    wav: torch.Tensor,
    mel_transform: torchaudio.transforms.MelSpectrogram,
    eps: float = 1.0e-10,
) -> torch.Tensor:
    spec = mel_transform(wav).squeeze(0).clamp_min(eps)
    flat = spec.reshape(-1)
    return flat / flat.sum().clamp_min(eps)


def log_mel(
    wav: torch.Tensor,
    mel_transform: torchaudio.transforms.MelSpectrogram,
    eps: float = 1.0e-10,
) -> torch.Tensor:
    return torch.log(mel_transform(wav).clamp_min(eps))


def spectral_metrics(
    reference: torch.Tensor,
    estimate: torch.Tensor,
    mel_transform: torchaudio.transforms.MelSpectrogram,
) -> dict[str, float]:
    reference, estimate = trim_pair(reference, estimate)
    ref_log = log_mel(reference, mel_transform)
    est_log = log_mel(estimate, mel_transform)
    p = mel_distribution(reference, mel_transform)
    q = mel_distribution(estimate, mel_transform)
    kl_ref_to_est = torch.sum(p * (torch.log(p) - torch.log(q)))
    kl_est_to_ref = torch.sum(q * (torch.log(q) - torch.log(p)))
    return {
        "log_mel_l1": float(torch.mean(torch.abs(ref_log - est_log))),
        "log_mel_mse": float(torch.mean(torch.square(ref_log - est_log))),
        "mel_kl_ref_to_est": float(kl_ref_to_est),
        "mel_kl_est_to_ref": float(kl_est_to_ref),
        "mel_symmetric_kl": float(0.5 * (kl_ref_to_est + kl_est_to_ref)),
    }


def summarize(rows: list[dict[str, Any]], group_key: str) -> dict[str, dict[str, dict[str, float]]]:
    grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        group = str(row[group_key])
        for key, value in row.items():
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                grouped[group][key].append(float(value))

    summary: dict[str, dict[str, dict[str, float]]] = {}
    for group, values_by_key in grouped.items():
        summary[group] = {}
        for key, values in values_by_key.items():
            stats = {
                "mean": statistics.fmean(values),
                "median": statistics.median(values),
                "min": min(values),
                "max": max(values),
            }
            if len(values) > 1:
                stats["stdev"] = statistics.stdev(values)
            summary[group][key] = stats
    return summary


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def save_metric_plots(eval_rows: list[dict[str, Any]], analysis_dir: Path) -> list[str]:
    plots: list[str] = []
    direct = [float(row["direct_overlap_si_sdr_mean"]) for row in eval_rows]
    residual = [float(row["residual_overlap_si_sdr_mean"]) for row in eval_rows]
    delta = [float(row["residual_minus_direct_overlap_si_sdr"]) for row in eval_rows]

    plt.figure(figsize=(8, 5))
    plt.hist(direct, bins=30, alpha=0.65, label="direct")
    plt.hist(residual, bins=30, alpha=0.65, label="residual-derived")
    plt.xlabel("Overlap SI-SDR (dB)")
    plt.ylabel("Examples")
    plt.legend()
    plt.tight_layout()
    path = analysis_dir / "overlap_si_sdr_hist.png"
    plt.savefig(path, dpi=160)
    plt.close()
    plots.append(path.name)

    plt.figure(figsize=(8, 5))
    plt.hist(delta, bins=30, color="#4c78a8")
    plt.axvline(0.0, color="black", linewidth=1)
    plt.xlabel("Residual minus direct overlap SI-SDR (dB)")
    plt.ylabel("Examples")
    plt.tight_layout()
    path = analysis_dir / "residual_minus_direct_overlap_si_sdr_hist.png"
    plt.savefig(path, dpi=160)
    plt.close()
    plots.append(path.name)

    lo = min(min(direct), min(residual))
    hi = max(max(direct), max(residual))
    plt.figure(figsize=(6, 6))
    plt.scatter(direct, residual, s=18, alpha=0.75)
    plt.plot([lo, hi], [lo, hi], color="black", linewidth=1)
    plt.xlabel("Direct overlap SI-SDR (dB)")
    plt.ylabel("Residual-derived overlap SI-SDR (dB)")
    plt.tight_layout()
    path = analysis_dir / "direct_vs_residual_overlap_si_sdr_scatter.png"
    plt.savefig(path, dpi=160)
    plt.close()
    plots.append(path.name)
    return plots


def save_spectral_plots(rows: list[dict[str, Any]], analysis_dir: Path) -> list[str]:
    plots: list[str] = []
    for metric, filename, xlabel in (
        ("overlap_mel_symmetric_kl", "overlap_mel_symmetric_kl_hist.png", "Overlap mel symmetric KL"),
        ("overlap_log_mel_l1", "overlap_log_mel_l1_hist.png", "Overlap log-mel L1"),
        ("full_mel_symmetric_kl", "full_mel_symmetric_kl_hist.png", "Full mel symmetric KL"),
    ):
        direct = [float(row[metric]) for row in rows if row["mode"] == "direct" and metric in row]
        residual = [
            float(row[metric]) for row in rows if row["mode"] == "residual" and metric in row
        ]
        if not direct or not residual:
            continue
        plt.figure(figsize=(8, 5))
        plt.hist(direct, bins=30, alpha=0.65, label="direct")
        plt.hist(residual, bins=30, alpha=0.65, label="residual-derived")
        plt.xlabel(xlabel)
        plt.ylabel("Speaker reconstructions")
        plt.legend()
        plt.tight_layout()
        path = analysis_dir / filename
        plt.savefig(path, dpi=160)
        plt.close()
        plots.append(path.name)
    return plots


def write_html_report(
    analysis_dir: Path,
    metric_plots: list[str],
    spectral_plots: list[str],
    spectral_summary: dict[str, dict[str, dict[str, float]]],
) -> None:
    summary_rows = []
    for group in sorted(spectral_summary):
        metrics = spectral_summary[group]
        summary_rows.append(
            "<tr>"
            f"<td>{html.escape(group)}</td>"
            f"<td>{metrics['overlap_mel_symmetric_kl']['mean']:.4f}</td>"
            f"<td>{metrics['overlap_log_mel_l1']['mean']:.4f}</td>"
            f"<td>{metrics['full_mel_symmetric_kl']['mean']:.4f}</td>"
            f"<td>{metrics['full_log_mel_l1']['mean']:.4f}</td>"
            "</tr>"
        )
    images = "\n".join(
        f'<figure><img src="{html.escape(name)}" alt="{html.escape(name)}"><figcaption>{html.escape(name)}</figcaption></figure>'
        for name in metric_plots + spectral_plots
    )
    report = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>SAM Audio Separation Analysis</title>
  <style>
    body {{
      color: #141414;
      font: 14px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      margin: 32px;
      max-width: 1100px;
    }}
    img {{ max-width: 100%; }}
    figure {{ margin: 24px 0; }}
    table {{ border-collapse: collapse; margin: 16px 0 28px; width: 100%; }}
    th, td {{ border: 1px solid #d8d8d8; padding: 6px 8px; text-align: left; }}
    th {{ background: #f3f3f3; }}
  </style>
</head>
<body>
  <h1>SAM Audio Separation Analysis</h1>
  <p>
    Mel KL here means KL between normalized mel-spectrogram energy
    distributions, not likelihood under a probabilistic model.
    Lower is better for all mel metrics.
  </p>
  <table>
    <thead>
      <tr>
        <th>Reconstruction</th>
        <th>Mean overlap symmetric KL</th>
        <th>Mean overlap log-mel L1</th>
        <th>Mean full symmetric KL</th>
        <th>Mean full log-mel L1</th>
      </tr>
    </thead>
    <tbody>
      {''.join(summary_rows)}
    </tbody>
  </table>
  {images}
</body>
</html>
"""
    (analysis_dir / "index.html").write_text(report)


def main() -> None:
    args = parse_args()
    analysis_dir = args.output_dir or args.eval_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    eval_rows = read_jsonl(args.eval_dir / "results.jsonl")
    manifest = load_manifest(args.manifest)
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=args.sample_rate,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        n_mels=args.n_mels,
        power=2.0,
    )

    spectral_rows: list[dict[str, Any]] = []
    for result in eval_rows:
        mixture_id = result["mixture_id"]
        sample_dir = args.eval_dir / mixture_id
        overlap_start = manifest.get(mixture_id, {}).get("overlap_start")
        overlap_end = manifest.get(mixture_id, {}).get("overlap_end")
        for label, ref_name, estimate_name, mode in EVAL_PAIRS:
            reference = load_audio(sample_dir / ref_name, args.sample_rate)
            estimate = load_audio(sample_dir / estimate_name, args.sample_rate)
            row: dict[str, Any] = {
                "mixture_id": mixture_id,
                "reconstruction": label,
                "mode": mode,
            }
            for key, value in spectral_metrics(reference, estimate, mel_transform).items():
                row[f"full_{key}"] = value
            if overlap_start is not None and overlap_end is not None:
                ref_overlap = slice_seconds(
                    reference, float(overlap_start), float(overlap_end), args.sample_rate
                )
                est_overlap = slice_seconds(
                    estimate, float(overlap_start), float(overlap_end), args.sample_rate
                )
                for key, value in spectral_metrics(
                    ref_overlap, est_overlap, mel_transform
                ).items():
                    row[f"overlap_{key}"] = value
            spectral_rows.append(row)

    write_csv(analysis_dir / "spectral_metrics.csv", spectral_rows)
    (analysis_dir / "spectral_metrics.jsonl").write_text(
        "\n".join(json.dumps(row) for row in spectral_rows) + "\n"
    )
    spectral_summary = summarize(spectral_rows, "reconstruction")
    (analysis_dir / "spectral_summary.json").write_text(
        json.dumps(spectral_summary, indent=2) + "\n"
    )

    metric_plots = save_metric_plots(eval_rows, analysis_dir)
    spectral_plots = save_spectral_plots(spectral_rows, analysis_dir)
    write_html_report(analysis_dir, metric_plots, spectral_plots, spectral_summary)
    print(f"Wrote analysis to {analysis_dir}")


if __name__ == "__main__":
    main()
