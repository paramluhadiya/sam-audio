"""Post-train SAM Audio on synthetic two-speaker span separation."""

from __future__ import annotations

import argparse
import html
import json
import math
import os
import shutil
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

from sam_audio import SAMAudio, SAMAudioProcessor
from run_span_eval import (
    flatten_metrics,
    load_audio,
    metric_block,
    read_manifest,
    run_one_prompt,
    save_audio,
    summarize,
)
from training_data import build_training_tensors


class SpanSeparationDataset(Dataset):
    def __init__(self, manifest: Path):
        self.rows: list[dict[str, Any]] = []
        with manifest.open() as handle:
            for line in handle:
                if line.strip():
                    self.rows.append(json.loads(line))
        if not self.rows:
            raise RuntimeError(f"No examples found in {manifest}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.rows[index]
        mix, target, residual = build_training_tensors(record)
        return {
            "mix": mix,
            "target": target,
            "residual": residual,
            "anchor": tuple(record["anchor"]),
            "target_source": int(record["target_source"]),
            "description": record.get("description", ""),
            "example_id": record["example_id"],
        }


def collate_examples(examples: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "mixes": [example["mix"] for example in examples],
        "targets": torch.stack([example["target"] for example in examples]),
        "residuals": torch.stack([example["residual"] for example in examples]),
        "anchors": [[example["anchor"]] for example in examples],
        "target_sources": torch.tensor(
            [example["target_source"] for example in examples], dtype=torch.long
        ),
        "descriptions": [example["description"] for example in examples],
        "example_ids": [example["example_id"] for example in examples],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1.0e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--target-loss-weight", type=float, default=1.0)
    parser.add_argument("--residual-loss-weight", type=float, default=1.0)
    parser.add_argument("--log-every-steps", type=int, default=10)
    parser.add_argument("--save-every-steps", type=int, default=100)
    parser.add_argument("--keep-last-checkpoints", type=int, default=2)
    parser.add_argument("--resume-checkpoint", type=Path, default=None)
    parser.add_argument(
        "--save-full-model",
        action="store_true",
        help="Save every model parameter. Default saves only trainable parameters.",
    )
    parser.add_argument(
        "--save-optimizer",
        action="store_true",
        help="Include optimizer state in checkpoints. This is large for full fine-tuning.",
    )
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--wandb-mode", default="online")
    parser.add_argument("--eval-manifest", type=Path, default=None)
    parser.add_argument("--eval-output-dir", type=Path, default=None)
    parser.add_argument("--eval-every-steps", type=int, default=0)
    parser.add_argument("--eval-limit", type=int, default=4)
    parser.add_argument("--eval-candidates", type=int, default=1)
    parser.add_argument("--eval-audio-examples", type=int, default=2)
    parser.add_argument(
        "--eval-save-audio-examples",
        type=int,
        default=4,
        help="Save local WAV/HTML audio panels for this many eval rows. Metrics still use --eval-limit rows.",
    )
    parser.add_argument(
        "--eval-at-start",
        action="store_true",
        help="Run one held-out separation eval before the first optimizer step.",
    )
    parser.add_argument(
        "--eval-save-residuals",
        action="store_true",
        help="Also save raw residual WAVs from each eval prompt.",
    )
    parser.add_argument(
        "--train-all",
        action="store_true",
        help="Fine-tune every SAM Audio parameter. By default only the flow model and prompt adapters train.",
    )
    parser.add_argument(
        "--dtype",
        choices=("float32", "bfloat16"),
        default="bfloat16",
        help="Autocast dtype used on CUDA.",
    )
    return parser.parse_args()


def set_trainable(model: SAMAudio, train_all: bool) -> None:
    if train_all:
        for parameter in model.parameters():
            parameter.requires_grad = True
        return

    for parameter in model.parameters():
        parameter.requires_grad = False
    for module in (
        model.transformer,
        model.proj,
        model.embed_anchors,
        model.memory_proj,
    ):
        for parameter in module.parameters():
            parameter.requires_grad = True


def set_frozen_modules_eval(model: SAMAudio) -> None:
    model.audio_codec.eval()
    model.text_encoder.eval()
    model.vision_encoder.eval()


def serialize_args(args: argparse.Namespace) -> dict[str, Any]:
    serialized: dict[str, Any] = {}
    for key, value in vars(args).items():
        serialized[key] = str(value) if isinstance(value, Path) else value
    return serialized


def maybe_init_wandb(args: argparse.Namespace):
    if args.wandb_project is None:
        return None
    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError(
            "wandb logging was requested, but wandb is not installed. "
            "Run `pip install wandb` or omit --wandb-project."
        ) from exc
    return wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_run_name,
        mode=args.wandb_mode,
        config=serialize_args(args),
    )


def encode_endpoint(
    model: SAMAudio,
    target: torch.Tensor,
    residual: torch.Tensor,
) -> torch.Tensor:
    target_features = model.audio_codec(target).transpose(1, 2)
    residual_features = model.audio_codec(residual).transpose(1, 2)
    return torch.cat([target_features, residual_features], dim=2)


def crop_forward_args(
    forward_args: dict[str, torch.Tensor],
    endpoint: torch.Tensor,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    audio_features = forward_args["audio_features"]
    audio_pad_mask = forward_args["audio_pad_mask"]
    time_steps = min(endpoint.size(1), audio_features.size(1), audio_pad_mask.size(1))
    endpoint = endpoint[:, :time_steps]
    forward_args["audio_features"] = audio_features[:, :time_steps]
    forward_args["anchor_alignment"] = forward_args["anchor_alignment"][:, :time_steps]
    forward_args["audio_pad_mask"] = audio_pad_mask[:, :time_steps]
    if forward_args["masked_video_features"] is not None:
        forward_args["masked_video_features"] = forward_args["masked_video_features"][
            :, :, :time_steps
        ]
    return forward_args, endpoint


def masked_stream_losses(
    pred_velocity: torch.Tensor,
    true_velocity: torch.Tensor,
    valid_mask: torch.Tensor,
    target_weight: float,
    residual_weight: float,
) -> tuple[torch.Tensor, dict[str, float], dict[str, torch.Tensor]]:
    channels = pred_velocity.size(-1) // 2
    mask = valid_mask.unsqueeze(-1).to(pred_velocity.dtype)
    per_sample_denom = (mask.sum(dim=(1, 2)) * channels).clamp_min(1.0)

    target_square = (
        (pred_velocity[..., :channels] - true_velocity[..., :channels]).square() * mask
    )
    residual_square = (
        (pred_velocity[..., channels:] - true_velocity[..., channels:]).square() * mask
    )
    target_per_sample = target_square.sum(dim=(1, 2)) / per_sample_denom
    residual_per_sample = residual_square.sum(dim=(1, 2)) / per_sample_denom
    loss_per_sample = target_weight * target_per_sample + residual_weight * residual_per_sample
    target_loss = target_per_sample.mean()
    residual_loss = residual_per_sample.mean()
    loss = loss_per_sample.mean()
    return loss, {
        "target_loss": float(target_loss.detach().cpu()),
        "residual_loss": float(residual_loss.detach().cpu()),
        "loss": float(loss.detach().cpu()),
    }, {
        "target_loss": target_per_sample.detach(),
        "residual_loss": residual_per_sample.detach(),
        "loss": loss_per_sample.detach(),
    }


def checkpoint_model_state(model: SAMAudio, save_full_model: bool) -> dict[str, torch.Tensor]:
    if save_full_model:
        return {key: value.detach().cpu() for key, value in model.state_dict().items()}

    trainable_names = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    return {
        key: value.detach().cpu()
        for key, value in model.state_dict().items()
        if key in trainable_names
    }


def prune_checkpoints(output_dir: Path, keep_last: int) -> None:
    if keep_last <= 0:
        return
    checkpoints = sorted(
        path
        for path in output_dir.glob("step_*")
        if path.is_dir() and path.name.removeprefix("step_").isdigit()
    )
    for checkpoint_dir in checkpoints[:-keep_last]:
        shutil.rmtree(checkpoint_dir)


def save_checkpoint(
    output_dir: Path,
    step: int,
    model: SAMAudio,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
) -> None:
    checkpoint_dir = output_dir / f"step_{step:08d}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "step": step,
        "checkpoint_type": "full_model" if args.save_full_model else "trainable_only",
        "model": checkpoint_model_state(model, args.save_full_model),
        "args": serialize_args(args),
    }
    if args.save_optimizer:
        checkpoint["optimizer"] = optimizer.state_dict()
    torch.save(checkpoint, checkpoint_dir / "training_state.pt")
    latest = output_dir / "latest"
    tmp_latest = output_dir / "latest.tmp"
    if tmp_latest.exists() or tmp_latest.is_symlink():
        tmp_latest.unlink()
    tmp_latest.symlink_to(checkpoint_dir.name, target_is_directory=True)
    if latest.exists() or latest.is_symlink():
        latest.unlink()
    tmp_latest.rename(latest)
    prune_checkpoints(output_dir, args.keep_last_checkpoints)


def load_checkpoint(
    checkpoint_path: Path,
    model: SAMAudio,
    optimizer: torch.optim.Optimizer,
) -> int:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    missing, unexpected = model.load_state_dict(checkpoint["model"], strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected checkpoint keys: {unexpected}")
    skipped_missing = [name for name in missing if name not in checkpoint["model"]]
    if checkpoint.get("checkpoint_type") == "full_model" and skipped_missing:
        raise RuntimeError(f"Missing full-model checkpoint keys: {skipped_missing}")
    if "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    return int(checkpoint["step"])


def new_accum_stats() -> dict[str, Any]:
    return {
        "all": {"count": 0, "loss": 0.0, "target_loss": 0.0, "residual_loss": 0.0},
        "front_anchor": {
            "count": 0,
            "loss": 0.0,
            "target_loss": 0.0,
            "residual_loss": 0.0,
        },
        "end_anchor": {
            "count": 0,
            "loss": 0.0,
            "target_loss": 0.0,
            "residual_loss": 0.0,
        },
    }


def add_accum_stats(
    stats: dict[str, Any],
    per_sample_losses: dict[str, torch.Tensor],
    target_sources: torch.Tensor,
) -> None:
    target_sources = target_sources.detach().cpu()
    per_sample_cpu = {
        name: values.detach().float().cpu() for name, values in per_sample_losses.items()
    }
    for index, source in enumerate(target_sources.tolist()):
        side = "front_anchor" if int(source) == 1 else "end_anchor"
        for bucket_name in ("all", side):
            bucket = stats[bucket_name]
            bucket["count"] += 1
            for metric_name, values in per_sample_cpu.items():
                bucket[metric_name] += float(values[index])


def finalize_accum_stats(stats: dict[str, Any]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for bucket_name, bucket in stats.items():
        count = int(bucket["count"])
        metrics[f"{bucket_name}_count"] = float(count)
        if count == 0:
            continue
        prefix = "train" if bucket_name == "all" else f"train/{bucket_name}"
        for metric_name in ("loss", "target_loss", "residual_loss"):
            metrics[f"{prefix}/{metric_name}"] = float(bucket[metric_name]) / count
    if "train/loss" in metrics:
        metrics["loss"] = metrics["train/loss"]
        metrics["target_loss"] = metrics["train/target_loss"]
        metrics["residual_loss"] = metrics["train/residual_loss"]
    return metrics


def build_eval_summary(
    args: argparse.Namespace,
    step: int,
    sample_rate: int,
    result_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "manifest": str(args.eval_manifest),
        "checkpoint_path": args.checkpoint_path,
        "step": step,
        "num_examples": len(result_rows),
        "sample_rate": sample_rate,
        "prompt_mode": "span",
        "candidates": args.eval_candidates,
        "metrics": summarize(result_rows),
    }


def format_db(value: Any) -> str:
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        return "n/a"
    return f"{float(value):.2f} dB"


def mean_metric(summary: dict[str, Any], name: str) -> float | None:
    value = summary.get("metrics", {}).get(name, {}).get("mean")
    return float(value) if isinstance(value, (int, float)) else None


def write_training_eval_report(
    output_dir: Path,
    result_rows: list[dict[str, Any]],
    summary: dict[str, Any],
) -> None:
    saved_rows = [record for record in result_rows if record.get("audio_saved")]
    cards: list[str] = []
    for record in saved_rows:
        mixture_id = html.escape(str(record["mixture_id"]))
        cards.append(
            f"""
<section>
  <h2>{mixture_id}</h2>
  <p>
    Front direct overlap SI-SDR: {html.escape(format_db(record["front_anchor_direct_overlap_si_sdr"]))}.
    End direct overlap SI-SDR: {html.escape(format_db(record["end_anchor_direct_overlap_si_sdr"]))}.
  </p>
  <div class="audio-grid">
    <div><h3>Mixture</h3><audio controls src="{mixture_id}/mix.wav"></audio></div>
    <div><h3>Front Target Ref (Source 1)</h3><audio controls src="{mixture_id}/front_anchor/target_ref_source1.wav"></audio></div>
    <div><h3>Front Target Pred (Source 1)</h3><audio controls src="{mixture_id}/front_anchor/target_pred_source1.wav"></audio></div>
    <div><h3>Front Residual Ref (Source 2)</h3><audio controls src="{mixture_id}/front_anchor/residual_ref_source2.wav"></audio></div>
    <div><h3>Front Residual Pred (Source 2)</h3><audio controls src="{mixture_id}/front_anchor/residual_pred_source2.wav"></audio></div>
    <div><h3>End Target Ref (Source 2)</h3><audio controls src="{mixture_id}/end_anchor/target_ref_source2.wav"></audio></div>
    <div><h3>End Target Pred (Source 2)</h3><audio controls src="{mixture_id}/end_anchor/target_pred_source2.wav"></audio></div>
    <div><h3>End Residual Ref (Source 1)</h3><audio controls src="{mixture_id}/end_anchor/residual_ref_source1.wav"></audio></div>
    <div><h3>End Residual Pred (Source 1)</h3><audio controls src="{mixture_id}/end_anchor/residual_pred_source1.wav"></audio></div>
  </div>
</section>
"""
        )

    rows: list[str] = []
    for record in result_rows:
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(record['mixture_id']))}</td>"
            f"<td>{html.escape(format_db(record['front_anchor_direct_overlap_si_sdr']))}</td>"
            f"<td>{html.escape(format_db(record['end_anchor_direct_overlap_si_sdr']))}</td>"
            f"<td>{html.escape(format_db(record['front_anchor_residual_overlap_si_sdr']))}</td>"
            f"<td>{html.escape(format_db(record['end_anchor_residual_overlap_si_sdr']))}</td>"
            "</tr>"
        )

    report = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>SAM Audio Training Separation Eval</title>
  <style>
    body {{
      color: #141414;
      font: 14px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      margin: 32px;
      max-width: 1180px;
    }}
    h1, h2, h3 {{ margin: 0 0 8px; }}
    h1 {{ font-size: 24px; }}
    h2 {{ font-size: 18px; margin-top: 28px; }}
    h3 {{ font-size: 13px; }}
    p {{ margin: 0 0 14px; }}
    section {{ border-top: 1px solid #d8d8d8; padding-top: 20px; }}
    audio {{ width: 100%; }}
    table {{ border-collapse: collapse; margin: 18px 0 24px; width: 100%; }}
    th, td {{ border: 1px solid #d8d8d8; padding: 6px 8px; text-align: left; }}
    th {{ background: #f3f3f3; }}
    .audio-grid {{
      display: grid;
      gap: 12px;
      grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
    }}
  </style>
</head>
<body>
  <h1>SAM Audio Training Separation Eval</h1>
  <p>
    Examples: {summary["num_examples"]}.
    Front direct overlap SI-SDR: {html.escape(format_db(mean_metric(summary, "front_anchor_direct_overlap_si_sdr")))}.
    End direct overlap SI-SDR: {html.escape(format_db(mean_metric(summary, "end_anchor_direct_overlap_si_sdr")))}.
    Front residual overlap SI-SDR: {html.escape(format_db(mean_metric(summary, "front_anchor_residual_overlap_si_sdr")))}.
    End residual overlap SI-SDR: {html.escape(format_db(mean_metric(summary, "end_anchor_residual_overlap_si_sdr")))}.
  </p>
  <table>
    <thead>
      <tr>
        <th>Mixture</th>
        <th>Front Direct</th>
        <th>End Direct</th>
        <th>Front Residual</th>
        <th>End Residual</th>
      </tr>
    </thead>
    <tbody>
      {"".join(rows)}
    </tbody>
  </table>
  {"".join(cards)}
</body>
</html>
"""
    (output_dir / "index.html").write_text(report)


def save_eval_summary(
    args: argparse.Namespace,
    step_dir: Path,
    step: int,
    sample_rate: int,
    result_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    summary = build_eval_summary(args, step, sample_rate, result_rows)
    (step_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    write_training_eval_report(step_dir, result_rows, summary)
    return summary


def append_eval_log(output_dir: Path, step_dir: Path, summary: dict[str, Any]) -> None:
    record: dict[str, Any] = {
        "step": summary["step"],
        "eval_dir": str(step_dir),
        "num_examples": summary["num_examples"],
    }
    for name, values in summary["metrics"].items():
        mean = values.get("mean")
        if isinstance(mean, (int, float)) and math.isfinite(float(mean)):
            record[f"{name}_mean"] = float(mean)
    with (output_dir / "eval_log.jsonl").open("a") as handle:
        handle.write(json.dumps(record) + "\n")


def write_eval_root_index(output_root: Path) -> None:
    rows: list[str] = []
    for step_dir in sorted(output_root.glob("step_*")):
        summary_path = step_dir / "summary.json"
        report_path = step_dir / "index.html"
        if not summary_path.exists() or not report_path.exists():
            continue
        try:
            summary = json.loads(summary_path.read_text())
            metrics = summary.get("metrics", {})
            front_direct = metrics.get("front_anchor_direct_overlap_si_sdr", {}).get("mean")
            end_direct = metrics.get("end_anchor_direct_overlap_si_sdr", {}).get("mean")
            front_residual = metrics.get(
                "front_anchor_residual_overlap_si_sdr", {}
            ).get("mean")
            end_residual = metrics.get("end_anchor_residual_overlap_si_sdr", {}).get(
                "mean"
            )
        except (OSError, json.JSONDecodeError):
            front_direct = end_direct = front_residual = end_residual = None
        metric_cells = [
            format_db(value)
            for value in (front_direct, end_direct, front_residual, end_residual)
        ]
        step_name = html.escape(step_dir.name)
        rows.append(
            f"<tr><td><a href=\"{step_name}/index.html\">{step_name}</a></td>"
            f"<td>{html.escape(metric_cells[0])}</td>"
            f"<td>{html.escape(metric_cells[1])}</td>"
            f"<td>{html.escape(metric_cells[2])}</td>"
            f"<td>{html.escape(metric_cells[3])}</td></tr>"
        )

    body = "\n".join(rows)
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "index.html").write_text(
        f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>SAM Audio Training Eval Timeline</title>
  <style>
    body {{
      color: #141414;
      font: 14px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      margin: 32px;
      max-width: 920px;
    }}
    h1 {{ font-size: 24px; margin: 0 0 16px; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border: 1px solid #d8d8d8; padding: 6px 8px; text-align: left; }}
    th {{ background: #f3f3f3; }}
  </style>
</head>
<body>
  <h1>SAM Audio Training Eval Timeline</h1>
  <table>
    <thead>
      <tr>
        <th>Step</th>
        <th>Front Direct</th>
        <th>End Direct</th>
        <th>Front Residual</th>
        <th>End Residual</th>
      </tr>
    </thead>
    <tbody>
      {body}
    </tbody>
  </table>
</body>
</html>
"""
    )


def update_latest_eval(output_root: Path, step_dir: Path) -> None:
    latest = output_root / "latest"
    tmp_latest = output_root / "latest.tmp"
    if tmp_latest.exists() or tmp_latest.is_symlink():
        tmp_latest.unlink()
    tmp_latest.symlink_to(step_dir.name, target_is_directory=True)
    if latest.exists() or latest.is_symlink():
        latest.unlink()
    tmp_latest.rename(latest)
    write_eval_root_index(output_root)


def log_eval_to_wandb(
    wandb_run: Any,
    step: int,
    step_dir: Path,
    sample_rate: int,
    result_rows: list[dict[str, Any]],
    summary: dict[str, Any],
    audio_examples: int,
) -> None:
    if wandb_run is None:
        return

    scalar_payload: dict[str, float] = {"eval/num_examples": float(len(result_rows))}
    for name, values in summary["metrics"].items():
        mean = values.get("mean")
        if isinstance(mean, (int, float)) and math.isfinite(float(mean)):
            scalar_payload[f"eval/{name}"] = float(mean)
    wandb_run.log(scalar_payload, step=step)

    if audio_examples <= 0:
        return

    import wandb

    audio_payload: dict[str, Any] = {}
    audio_files = (
        ("mix", "mix.wav"),
        ("front_anchor/target_ref_source1", "front_anchor/target_ref_source1.wav"),
        ("front_anchor/target_pred_source1", "front_anchor/target_pred_source1.wav"),
        (
            "front_anchor/residual_ref_source2",
            "front_anchor/residual_ref_source2.wav",
        ),
        (
            "front_anchor/residual_pred_source2",
            "front_anchor/residual_pred_source2.wav",
        ),
        ("end_anchor/target_ref_source2", "end_anchor/target_ref_source2.wav"),
        ("end_anchor/target_pred_source2", "end_anchor/target_pred_source2.wav"),
        ("end_anchor/residual_ref_source1", "end_anchor/residual_ref_source1.wav"),
        ("end_anchor/residual_pred_source1", "end_anchor/residual_pred_source1.wav"),
    )
    for record in result_rows[:audio_examples]:
        if not record.get("audio_saved"):
            continue
        mixture_id = str(record["mixture_id"])
        sample_dir = step_dir / mixture_id
        for label, filename in audio_files:
            path = sample_dir / filename
            if path.exists():
                audio_payload[f"eval_audio/{mixture_id}/{label}"] = wandb.Audio(
                    str(path),
                    sample_rate=sample_rate,
                    caption=f"step {step} {mixture_id} {label}",
                )
    if audio_payload:
        wandb_run.log(audio_payload, step=step)


def run_training_eval(
    args: argparse.Namespace,
    model: SAMAudio,
    processor: SAMAudioProcessor,
    device: torch.device,
    step: int,
    wandb_run: Any,
) -> dict[str, Any] | None:
    if args.eval_manifest is None:
        return None

    output_root = args.eval_output_dir or args.output_dir / "eval"
    step_dir = output_root / f"step_{step:08d}"
    step_dir.mkdir(parents=True, exist_ok=True)
    rows = read_manifest(args.eval_manifest, args.eval_limit)
    if not rows:
        raise RuntimeError(f"No eval rows found in {args.eval_manifest}")

    sample_rate = processor.audio_sampling_rate
    result_rows: list[dict[str, Any]] = []
    results_jsonl = step_dir / "results.jsonl"
    was_training = model.training
    model.eval()
    try:
        with results_jsonl.open("w") as out:
            for index, record in enumerate(rows):
                mixture_id = record["mixture_id"]
                mixture_output_dir = step_dir / mixture_id
                save_example_audio = index < args.eval_save_audio_examples
                pred1, residual1 = run_one_prompt(
                    model=model,
                    processor=processor,
                    mix_path=record["mix_path"],
                    description="",
                    anchor=record["source1_anchor"],
                    device=device,
                    candidates=args.eval_candidates,
                )
                pred2, residual2 = run_one_prompt(
                    model=model,
                    processor=processor,
                    mix_path=record["mix_path"],
                    description="",
                    anchor=record["source2_anchor"],
                    device=device,
                    candidates=args.eval_candidates,
                )

                ref1 = load_audio(record["source1_path"], sample_rate)
                ref2 = load_audio(record["source2_path"], sample_rate)
                mix = load_audio(record["mix_path"], sample_rate)

                if save_example_audio:
                    save_audio(mixture_output_dir / "mix.wav", mix, sample_rate)
                    save_audio(
                        mixture_output_dir / "front_anchor/target_ref_source1.wav",
                        ref1,
                        sample_rate,
                    )
                    save_audio(
                        mixture_output_dir / "front_anchor/target_pred_source1.wav",
                        pred1,
                        sample_rate,
                    )
                    save_audio(
                        mixture_output_dir / "front_anchor/residual_ref_source2.wav",
                        ref2,
                        sample_rate,
                    )
                    save_audio(
                        mixture_output_dir / "front_anchor/residual_pred_source2.wav",
                        residual1,
                        sample_rate,
                    )
                    save_audio(
                        mixture_output_dir / "end_anchor/target_ref_source2.wav",
                        ref2,
                        sample_rate,
                    )
                    save_audio(
                        mixture_output_dir / "end_anchor/target_pred_source2.wav",
                        pred2,
                        sample_rate,
                    )
                    save_audio(
                        mixture_output_dir / "end_anchor/residual_ref_source1.wav",
                        ref1,
                        sample_rate,
                    )
                    save_audio(
                        mixture_output_dir / "end_anchor/residual_pred_source1.wav",
                        residual2,
                        sample_rate,
                    )
                    if args.eval_save_residuals:
                        save_audio(
                            mixture_output_dir
                            / "front_anchor/raw_residual_from_front_prompt.wav",
                            residual1,
                            sample_rate,
                        )
                        save_audio(
                            mixture_output_dir / "end_anchor/raw_residual_from_end_prompt.wav",
                            residual2,
                            sample_rate,
                        )

                overlap_start = float(record["overlap_start"])
                overlap_end = float(record["overlap_end"])
                front_anchor_direct = metric_block(
                    pred1, ref1, mix, sample_rate, overlap_start, overlap_end
                )
                end_anchor_direct = metric_block(
                    pred2, ref2, mix, sample_rate, overlap_start, overlap_end
                )
                end_anchor_residual = metric_block(
                    residual2, ref1, mix, sample_rate, overlap_start, overlap_end
                )
                front_anchor_residual = metric_block(
                    residual1, ref2, mix, sample_rate, overlap_start, overlap_end
                )
                front_direct_swapped = metric_block(
                    pred1, ref2, mix, sample_rate, overlap_start, overlap_end
                )
                end_direct_swapped = metric_block(
                    pred2, ref1, mix, sample_rate, overlap_start, overlap_end
                )
                direct_overlap = (
                    front_anchor_direct["overlap_si_sdr"]
                    + end_anchor_direct["overlap_si_sdr"]
                ) / 2.0
                swapped_overlap = (
                    front_direct_swapped["overlap_si_sdr"]
                    + end_direct_swapped["overlap_si_sdr"]
                ) / 2.0
                residual_overlap = (
                    end_anchor_residual["overlap_si_sdr"]
                    + front_anchor_residual["overlap_si_sdr"]
                ) / 2.0

                result_record: dict[str, Any] = {
                    "index": index,
                    "mixture_id": mixture_id,
                    "speaker1_id": record["speaker1_id"],
                    "speaker2_id": record["speaker2_id"],
                    "prompt_mode": "span",
                    "audio_saved": save_example_audio,
                    "direct_overlap_si_sdr_mean": direct_overlap,
                    "residual_overlap_si_sdr_mean": residual_overlap,
                    "swapped_overlap_si_sdr_mean": swapped_overlap,
                    "assignment_correct": direct_overlap >= swapped_overlap,
                    "residual_minus_direct_overlap_si_sdr": residual_overlap
                    - direct_overlap,
                }
                for metric_name in ("si_sdr", "snr", "overlap_si_sdr", "overlap_snr"):
                    result_record[f"front_anchor_residual_minus_direct_{metric_name}"] = (
                        front_anchor_residual[metric_name]
                        - front_anchor_direct[metric_name]
                    )
                    result_record[f"end_anchor_residual_minus_direct_{metric_name}"] = (
                        end_anchor_residual[metric_name] - end_anchor_direct[metric_name]
                    )

                result_record.update(
                    flatten_metrics("front_anchor_direct", front_anchor_direct)
                )
                result_record.update(
                    flatten_metrics("front_anchor_residual", front_anchor_residual)
                )
                result_record.update(
                    flatten_metrics("end_anchor_direct", end_anchor_direct)
                )
                result_record.update(
                    flatten_metrics("end_anchor_residual", end_anchor_residual)
                )
                result_rows.append(result_record)
                out.write(json.dumps(result_record) + "\n")
                out.flush()
                os.fsync(out.fileno())
                save_eval_summary(args, step_dir, step, sample_rate, result_rows)

                print(
                    f"eval step {step} [{index + 1}/{len(rows)}] {mixture_id}: "
                    f"front direct={front_anchor_direct['overlap_si_sdr']:.2f} dB, "
                    f"end direct={end_anchor_direct['overlap_si_sdr']:.2f} dB, "
                    f"front residual={front_anchor_residual['overlap_si_sdr']:.2f} dB, "
                    f"end residual={end_anchor_residual['overlap_si_sdr']:.2f} dB"
                )
    finally:
        if was_training:
            model.train()
            set_frozen_modules_eval(model)

    summary = save_eval_summary(args, step_dir, step, sample_rate, result_rows)
    update_latest_eval(output_root, step_dir)
    append_eval_log(args.output_dir, step_dir, summary)
    log_eval_to_wandb(
        wandb_run=wandb_run,
        step=step,
        step_dir=step_dir,
        sample_rate=sample_rate,
        result_rows=result_rows,
        summary=summary,
        audio_examples=args.eval_audio_examples,
    )
    print(f"wrote eval step {step} report to {step_dir / 'index.html'}")
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return summary


def train() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "args.json").write_text(
        json.dumps(serialize_args(args), indent=2) + "\n"
    )
    wandb_run = maybe_init_wandb(args)

    model = SAMAudio.from_pretrained(
        args.checkpoint_path,
        text_ranker=None,
        visual_ranker=None,
    )
    processor = SAMAudioProcessor.from_pretrained(args.checkpoint_path)
    model = model.to(device)
    set_trainable(model, args.train_all)
    model.train()
    set_frozen_modules_eval(model)

    trainable_params = [param for param in model.parameters() if param.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    step = 0
    if args.resume_checkpoint is not None:
        step = load_checkpoint(args.resume_checkpoint, model, optimizer)
        print(f"resumed from {args.resume_checkpoint} at step {step}")
    dataset = SpanSeparationDataset(args.manifest)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_examples,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )

    autocast_dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    use_autocast = device.type == "cuda" and args.dtype != "float32"
    optimizer.zero_grad(set_to_none=True)
    log_path = args.output_dir / "train_log.jsonl"
    shutil.copy2(args.manifest, args.output_dir / "train_manifest.jsonl")
    accum_stats = new_accum_stats()
    accum_microbatches = 0
    last_eval_step: int | None = None
    if args.eval_at_start and args.eval_manifest is not None:
        run_training_eval(args, model, processor, device, step, wandb_run)
        last_eval_step = step

    with log_path.open("a") as log_file:
        for epoch in range(args.epochs):
            for batch_index, batch_data in enumerate(loader):
                batch = processor(
                    descriptions=batch_data["descriptions"],
                    audios=batch_data["mixes"],
                    anchors=batch_data["anchors"],
                ).to(device)
                targets = batch_data["targets"].to(device, non_blocking=True)
                residuals = batch_data["residuals"].to(device, non_blocking=True)
                target_sources = batch_data["target_sources"]

                with torch.autocast(
                    device_type=device.type,
                    dtype=autocast_dtype,
                    enabled=use_autocast,
                ):
                    endpoint = encode_endpoint(model, targets, residuals)
                    forward_args = model._get_forward_args(batch)
                    forward_args, endpoint = crop_forward_args(forward_args, endpoint)
                    noise = torch.randn_like(endpoint)
                    time = torch.rand(endpoint.size(0), device=device)
                    view_shape = (endpoint.size(0),) + (1,) * (endpoint.ndim - 1)
                    time_view = time.view(view_shape)
                    noisy_audio = (1.0 - time_view) * noise + time_view * endpoint
                    true_velocity = endpoint - noise
                    pred_velocity = model.forward(
                        noisy_audio=noisy_audio,
                        time=time,
                        **forward_args,
                    )
                    loss, loss_values, per_sample_losses = masked_stream_losses(
                        pred_velocity=pred_velocity,
                        true_velocity=true_velocity,
                        valid_mask=forward_args["audio_pad_mask"],
                        target_weight=args.target_loss_weight,
                        residual_weight=args.residual_loss_weight,
                    )
                    scaled_loss = loss / args.grad_accum_steps

                scaled_loss.backward()
                add_accum_stats(accum_stats, per_sample_losses, target_sources)
                accum_microbatches += 1
                if accum_microbatches < args.grad_accum_steps:
                    continue

                if args.max_grad_norm > 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        trainable_params, args.max_grad_norm
                    )
                    grad_norm_value = float(grad_norm.detach().cpu())
                else:
                    grad_norm_value = math.nan
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                accum_values = finalize_accum_stats(accum_stats)
                accum_stats = new_accum_stats()
                accum_microbatches = 0

                log_record = {
                    "step": step,
                    "epoch": epoch,
                    "batch_index": batch_index,
                    "lr": optimizer.param_groups[0]["lr"],
                    "grad_norm": grad_norm_value,
                    "last_microbatch_loss": loss_values["loss"],
                    "last_microbatch_target_loss": loss_values["target_loss"],
                    "last_microbatch_residual_loss": loss_values["residual_loss"],
                    **accum_values,
                }
                log_file.write(json.dumps(log_record) + "\n")
                log_file.flush()
                if wandb_run is not None:
                    wandb_payload = {
                        key: value
                        for key, value in accum_values.items()
                        if key.startswith("train/")
                    }
                    wandb_payload.update(
                        {
                            "train/lr": optimizer.param_groups[0]["lr"],
                            "train/epoch": epoch,
                            "train/batch_index": batch_index,
                            "train/grad_norm": grad_norm_value,
                            "train/last_microbatch_loss": loss_values["loss"],
                            "train/front_anchor_count": accum_values.get(
                                "front_anchor_count", 0.0
                            ),
                            "train/end_anchor_count": accum_values.get(
                                "end_anchor_count", 0.0
                            ),
                        }
                    )
                    wandb_run.log(wandb_payload, step=step)

                if step % args.log_every_steps == 0:
                    print(
                        f"step {step}: loss={accum_values['loss']:.4f}, "
                        f"target={accum_values['target_loss']:.4f}, "
                        f"residual={accum_values['residual_loss']:.4f}, "
                        f"front={accum_values.get('train/front_anchor/loss', math.nan):.4f}, "
                        f"end={accum_values.get('train/end_anchor/loss', math.nan):.4f}"
                    )
                if step % args.save_every_steps == 0:
                    save_checkpoint(args.output_dir, step, model, optimizer, args)
                    if wandb_run is not None:
                        wandb_run.log(
                            {"checkpoint/step": step, "checkpoint/saved": 1},
                            step=step,
                        )
                    print(f"saved checkpoint at step {step}")
                if (
                    args.eval_manifest is not None
                    and args.eval_every_steps > 0
                    and step % args.eval_every_steps == 0
                ):
                    run_training_eval(args, model, processor, device, step, wandb_run)
                    last_eval_step = step
                if args.max_steps is not None and step >= args.max_steps:
                    if args.eval_manifest is not None and last_eval_step != step:
                        run_training_eval(
                            args, model, processor, device, step, wandb_run
                        )
                    save_checkpoint(args.output_dir, step, model, optimizer, args)
                    if wandb_run is not None:
                        wandb_run.finish()
                    return

    if step == 0:
        raise RuntimeError("No optimizer steps were run")
    if step % args.save_every_steps != 0:
        save_checkpoint(args.output_dir, step, model, optimizer, args)
    if args.eval_manifest is not None and last_eval_step != step:
        run_training_eval(args, model, processor, device, step, wandb_run)
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    train()
