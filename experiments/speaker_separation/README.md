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
  --num-examples 50000 \
  --clip-duration 10.0 \
  --prompt-duration 1.0 \
  --sample-rate 48000 \
  --anchor-min-rms-dbfs -35 \
  --anchor-min-peak-dbfs -25 \
  --max-overlap-rms-delta-db 8 \
  --seed 13
```

The training manifest is lightweight: it stores source paths, source offsets,
speaker IDs, the chosen target side, the span anchor, filter stats, and
normalization/headroom settings. It does not precompute every mixture WAV.
The builder also writes a small `previews/` directory for spot checks.

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
  --output-dir /workspace/experiments/speaker-separation/train-small-v1/checkpoints \
  --device cuda \
  --batch-size 2 \
  --grad-accum-steps 8 \
  --learning-rate 1e-5 \
  --save-every-steps 100 \
  --log-every-steps 10
```

To enable wandb, export `WANDB_API_KEY` on the pod and add the tracking flags:

```bash
  --wandb-project sam-audio-speaker-separation \
  --wandb-run-name small-v1
```

By default the trainer freezes the codec, text encoder, vision encoder, rankers,
and span predictor, and fine-tunes the flow transformer plus the small prompt
conditioning adapters. Use `--train-all` only for a deliberately larger
fine-tuning run.

Omit `--wandb-project` for a local-only run that writes only `train_log.jsonl`
and checkpoints.
