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
