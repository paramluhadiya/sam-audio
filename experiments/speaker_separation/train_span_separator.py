"""Post-train SAM Audio on synthetic two-speaker span separation."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

from sam_audio import SAMAudio, SAMAudioProcessor
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
            "description": record.get("description", ""),
            "example_id": record["example_id"],
        }


def collate_examples(examples: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "mixes": [example["mix"] for example in examples],
        "targets": torch.stack([example["target"] for example in examples]),
        "residuals": torch.stack([example["residual"] for example in examples]),
        "anchors": [[example["anchor"]] for example in examples],
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
    parser.add_argument("--seed", type=int, default=13)
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


def serialize_args(args: argparse.Namespace) -> dict[str, Any]:
    serialized: dict[str, Any] = {}
    for key, value in vars(args).items():
        serialized[key] = str(value) if isinstance(value, Path) else value
    return serialized


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
) -> tuple[torch.Tensor, dict[str, float]]:
    channels = pred_velocity.size(-1) // 2
    mask = valid_mask.unsqueeze(-1).to(pred_velocity.dtype)
    denom = (mask.sum() * channels).clamp_min(1.0)

    target_loss = (
        (pred_velocity[..., :channels] - true_velocity[..., :channels]).square() * mask
    ).sum() / denom
    residual_loss = (
        (pred_velocity[..., channels:] - true_velocity[..., channels:]).square() * mask
    ).sum() / denom
    loss = target_weight * target_loss + residual_weight * residual_loss
    return loss, {
        "target_loss": float(target_loss.detach().cpu()),
        "residual_loss": float(residual_loss.detach().cpu()),
        "loss": float(loss.detach().cpu()),
    }


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
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "args": serialize_args(args),
    }
    torch.save(checkpoint, checkpoint_dir / "training_state.pt")
    latest = output_dir / "latest"
    tmp_latest = output_dir / "latest.tmp"
    if tmp_latest.exists() or tmp_latest.is_symlink():
        tmp_latest.unlink()
    tmp_latest.symlink_to(checkpoint_dir.name, target_is_directory=True)
    if latest.exists() or latest.is_symlink():
        latest.unlink()
    tmp_latest.rename(latest)


def train() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "args.json").write_text(
        json.dumps(serialize_args(args), indent=2) + "\n"
    )

    model = SAMAudio.from_pretrained(
        args.checkpoint_path,
        text_ranker=None,
        visual_ranker=None,
    )
    processor = SAMAudioProcessor.from_pretrained(args.checkpoint_path)
    model = model.to(device)
    set_trainable(model, args.train_all)
    model.train()
    model.audio_codec.eval()
    model.text_encoder.eval()
    model.vision_encoder.eval()

    trainable_params = [param for param in model.parameters() if param.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
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
    step = 0
    optimizer.zero_grad(set_to_none=True)
    log_path = args.output_dir / "train_log.jsonl"
    shutil.copy2(args.manifest, args.output_dir / "train_manifest.jsonl")

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
                    loss, loss_values = masked_stream_losses(
                        pred_velocity=pred_velocity,
                        true_velocity=true_velocity,
                        valid_mask=forward_args["audio_pad_mask"],
                        target_weight=args.target_loss_weight,
                        residual_weight=args.residual_loss_weight,
                    )
                    scaled_loss = loss / args.grad_accum_steps

                scaled_loss.backward()
                if (batch_index + 1) % args.grad_accum_steps != 0:
                    continue

                if args.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(trainable_params, args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1

                log_record = {
                    "step": step,
                    "epoch": epoch,
                    "batch_index": batch_index,
                    "lr": optimizer.param_groups[0]["lr"],
                    **loss_values,
                }
                log_file.write(json.dumps(log_record) + "\n")
                log_file.flush()

                if step % args.log_every_steps == 0:
                    print(
                        f"step {step}: loss={loss_values['loss']:.4f}, "
                        f"target={loss_values['target_loss']:.4f}, "
                        f"residual={loss_values['residual_loss']:.4f}"
                    )
                if step % args.save_every_steps == 0:
                    save_checkpoint(args.output_dir, step, model, optimizer, args)
                    print(f"saved checkpoint at step {step}")
                if args.max_steps is not None and step >= args.max_steps:
                    save_checkpoint(args.output_dir, step, model, optimizer, args)
                    return

    if step == 0:
        raise RuntimeError("No optimizer steps were run")
    if step % args.save_every_steps != 0:
        save_checkpoint(args.output_dir, step, model, optimizer, args)


if __name__ == "__main__":
    train()
