# Libri-Light Span Speaker Separation

This experiment tests whether SAM Audio span prompting can recover two known
speakers from synthetic mixtures.

## Mixture Layout

For each pair of single-speaker clips:

- source 1 is placed at `[0.0, clip_duration]`
- source 2 is placed at `[1.0, clip_duration + 1.0]` by default
- the first second contains only source 1
- the final second contains only source 2
- the middle region contains both speakers

The evaluation prompts SAM Audio twice:

- source 1 anchor: `[("+", 0.0, 1.0)]`
- source 2 anchor: `[("+", clip_duration, clip_duration + 1.0)]`

SAM Audio is run twice per mixture. For each speaker, we evaluate two
reconstructions:

- direct: the `target` returned when that speaker's solo span is prompted
- residual: the `residual` returned when the other speaker's solo span is prompted

Metrics are reported for the full generated clip and for the overlap-only
region. The overlap-only scores are the stricter numbers because they exclude
the solo prompt seconds. The residual-vs-direct deltas tell us whether a future
one-pass setup is enough.

## Pod Data Setup

Use the official Libri-Light limited-supervision bundle first. It is about
0.6 GB, while the unlabelled `small.tar` split is about 35 GB.

```bash
mkdir -p /workspace/data/libri-light
wget -c -P /workspace/data/libri-light \
  https://dl.fbaipublicfiles.com/librilight/data/librispeech_finetuning.tgz
```

On the current RunPod, `/workspace` is a persistent FUSE volume. Keep the
downloaded archive there, but stage extracted audio, mixtures, and hot inference
outputs under `/tmp/sam-audio` for faster local I/O:

```bash
mkdir -p /tmp/sam-audio/data/libri-light
tar --no-same-owner -xzf /workspace/data/libri-light/librispeech_finetuning.tgz \
  -C /tmp/sam-audio/data/libri-light
```

## Pod Environment Setup

Install SAM Audio first, then apply the pod-specific compatibility pins:

```bash
cd /tmp/sam-audio/repo
pip install .
apt-get update && apt-get install -y ffmpeg
pip install --upgrade -r experiments/speaker_separation/requirements-pod.txt
```

The extra requirements file pins a PyTorch/TorchCodec pair that imports cleanly
on the H100 pod. This matters because TorchCodec loads compiled libraries
against PyTorch's ABI, and SAM Audio imports `torchcodec.decoders.AudioDecoder`.

## Build Mixtures

Run this on the pod so the manifest contains pod-local absolute paths:

```bash
python experiments/speaker_separation/build_mixtures.py \
  --source-dir /tmp/sam-audio/data/libri-light \
  --output-dir /tmp/sam-audio/experiments/speaker-separation/mixtures \
  --num-mixtures 100 \
  --clip-duration 10.0 \
  --prompt-duration 1.0 \
  --sample-rate 48000 \
  --seed 13
```

The builder writes:

- `manifest.jsonl`: one row per mixture, including prompt anchors and reference paths
- `metadata.json`: scan and generation settings
- one directory per mixture containing `mix.wav`, `source1.wav`, and `source2.wav`

## Run Span Evaluation

Use the cached checkpoint snapshot directly:

```bash
python experiments/speaker_separation/run_span_eval.py \
  --manifest /tmp/sam-audio/experiments/speaker-separation/mixtures/manifest.jsonl \
  --checkpoint-path /workspace/hf-cache/models--facebook--sam-audio-large/snapshots/5f2cd3a9471a08c7282c06036be6893e18de8b70 \
  --output-dir /tmp/sam-audio/experiments/speaker-separation/span-eval \
  --device cuda \
  --prompt-mode span \
  --candidates 1
```

For a quick smoke test, add `--limit 2`. To include a speech text prompt in
addition to the span, use `--prompt-mode text-span`.

The evaluator writes each completed sample immediately. Every sample directory
contains the mixture, both ground-truth sources, the two direct reconstructions,
and the two residual-derived reconstructions. The top-level `index.html` is
refreshed after every sample and links to all listenable audio files with the
main per-sample metrics.

For training-time evals, prefer a held-out manifest whose front and end anchor
spans have both been checked for speech activity:

```bash
python experiments/speaker_separation/build_filtered_eval_mixtures.py \
  --source-dir /tmp/sam-audio/data/libri-light-small \
  --output-dir /workspace/experiments/speaker-separation/eval-filtered-v1 \
  --num-mixtures 64 \
  --clip-duration 10.0 \
  --prompt-duration 1.0 \
  --sample-rate 48000 \
  --anchor-min-rms-dbfs -45 \
  --anchor-min-peak-dbfs -55 \
  --seed 113
```

## Post-Training Data

For the first post-training run, use the official Libri-Light `small.tar` split
instead of the 10 hour limited-supervision bundle:

```bash
mkdir -p /workspace/data/libri-light
wget -c -O /workspace/data/libri-light/small.tar \
  https://dl.fbaipublicfiles.com/librilight/data/small.tar
mkdir -p /tmp/sam-audio/data/libri-light-small
tar --no-same-owner -xf /workspace/data/libri-light/small.tar \
  -C /tmp/sam-audio/data/libri-light-small
```

Build a filtered training manifest from the extracted audio:

```bash
python experiments/speaker_separation/build_training_manifest.py \
  --source-dir /tmp/sam-audio/data/libri-light-small \
  --output-dir /workspace/experiments/speaker-separation/train-small-v1 \
  --num-examples 5000 \
  --clip-duration 10.0 \
  --prompt-duration 1.0 \
  --sample-rate 48000 \
  --anchor-min-rms-dbfs -45 \
  --anchor-min-peak-dbfs -55 \
  --seed 13
```

The training manifest is lightweight: it stores source paths, source offsets,
speaker IDs, the chosen target side, the span anchor, filter stats, and
normalization/headroom settings. It does not precompute every mixture WAV.
The builder also writes a small `previews/` directory for spot checks.
By default it filters only the selected conditioning anchor, rejecting examples
where that one-second prompt is effectively silent. Add
`--max-overlap-rms-delta-db` for an optional overlap balance filter, or
`--require-both-anchors` to require both solo anchors to pass.

## Post-Training

The training objective samples examples where source 1 or source 2 has already
been selected as the prompted target with probability 1/2 by the manifest
builder. The flow endpoint is the concatenation of target-source codec latents
and residual-source codec latents:

```text
x_1 = concat(codec(target_source), codec(residual_source))
```

The standard flow-matching velocity loss is applied to both halves of that
endpoint. There is no extra mixture-consistency loss in this first version.

```bash
python experiments/speaker_separation/train_span_separator.py \
  --manifest /workspace/experiments/speaker-separation/train-small-v1/train_manifest.jsonl \
  --checkpoint-path /workspace/hf-cache/models--facebook--sam-audio-large/snapshots/5f2cd3a9471a08c7282c06036be6893e18de8b70 \
  --output-dir /workspace/experiments/speaker-separation/train-small-v2/checkpoints \
  --device cuda \
  --epochs 10 \
  --batch-size 1 \
  --grad-accum-steps 16 \
  --learning-rate 1e-5 \
  --save-every-steps 1000 \
  --log-every-steps 10 \
  --eval-manifest /workspace/experiments/speaker-separation/eval-filtered-v1/manifest.jsonl \
  --eval-output-dir /workspace/experiments/speaker-separation/train-small-v2/eval \
  --eval-every-steps 500 \
  --eval-limit 32 \
  --eval-save-audio-examples 2 \
  --eval-audio-examples 1 \
  --eval-at-start
```

To enable wandb, export `WANDB_API_KEY` on the pod and add the tracking flags:

```bash
  --wandb-project voice-separation \
  --wandb-run-name small-v1
```

By default the trainer freezes the codec, text encoder, vision encoder, rankers,
and span predictor, and fine-tunes the flow transformer plus the small prompt
conditioning adapters. Use `--train-all` only for a deliberately larger
fine-tuning run.

Checkpoints save trainable weights only by default and keep the last two
`step_*` directories, with `latest` pointing to the newest checkpoint. Add
`--save-optimizer` if optimizer-state resume is worth the extra disk, and
`--save-full-model` only if you explicitly want a standalone full-model state.

When `--eval-manifest` is supplied, training periodically runs the same
two-prompt held-out separation evaluation used by `run_span_eval.py`, but labels
the training-time results by anchor position. Each eval step writes:

- `results.jsonl` with per-mixture SI-SDR/SNR metrics split into
  `front_anchor_*` and `end_anchor_*` direct/residual reconstructions.
- `summary.json` with aggregate means/medians/min/max values.
- `index.html` with mixture, ground-truth, direct, and residual WAV players for
  only the first `--eval-save-audio-examples` rows.
- A top-level eval `index.html` timeline and `latest` symlink.
- Wandb scalar metrics and audio panels when `--wandb-project` is set.

Training loss logging is averaged over the full gradient-accumulation window,
not the last microbatch. The trainer logs aggregate `train/loss` plus
`train/front_anchor/*` and `train/end_anchor/*` losses so front-prompt and
end-prompt behavior can be diagnosed separately.

Omit `--wandb-project` for a local-only run that writes only `train_log.jsonl`
checkpoints, eval JSON, and eval WAV/HTML artifacts.
