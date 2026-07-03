# Turn-Taking Naturalness

Minimal code for generating five conversational timing perturbations, training
FVAD models, and scoring paired natural/edited conversations with NLL.

The public pipeline keeps generation and model evaluation separate. Generated
samples are produced from data-only timing and VAD rules; checkpoint evaluation
is run as a standalone post-training step.

## Setup

For data generation only:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For model training or checkpoint evaluation, install the optional model stack:

```bash
pip install -r requirements-train.txt
pip install -e VAP-main
```

Place a raw VAP model state dict from the upstream VAP repository at
`checkpoints/VAP_state_dict.pt` if you want to run VAP scoring or VAP training.
DualTurn checkpoints/backbones are loaded through Hugging Face unless you pass
local/offline paths.

## Generate Edited Data

Prepare a natural-conversation manifest using the format in
`data/manifests/README.md`. Each row should point to the two participant audio
files, and transcript/metadata paths if available.

Generate all five perturbation types:

```bash
python natural_classification/build_natural_unnatural_dualturn.py make-unnatural \
  --natural-csv data/manifests/test2_natural.csv \
  --out-root data/generated \
  --split test2 \
  --per-type 200 \
  --types early_entry,late_response,shift_instead_of_hold,hold_instead_of_shift,excessive_backchannel \
  --short-context
```

Outputs are written under `data/generated/`:

- `manifests/test2.csv`: paired natural/edited manifest
- `test2/audio/`: edited audio
- `test2/json/`: edited metadata, including `edit_meta`
- `test2/natural_audio/` and `test2/natural_json/`: matched natural references
- `generated_test2_rows.csv`, `test2_generation_log.csv`, and `test2_failures.csv`:
  generation bookkeeping and failure reasons

### Default Generation Logic

Common defaults:

- turn source: `silero`
- one edit per generated clip
- short-context crop target: `20-25s`
- crop guard keeps the edited boundary and nearby turn-taking context inside the crop
- generated WAVs are normalized to `-20 dBFS` RMS and `-1 dBFS` peak

Per-type defaults:

- `early_entry`: move the responder earlier by `1.2-2.5s` at a clean A-B turn
  transition.
- `late_response`: delay the responder by `1.2-2.0s`; require adjacent speaker
  and responder turns of at least `1s`, with at most `200ms` overlap.
- `shift_instead_of_hold`: insert a short shift turn into an original same-
  speaker hold gap of `0.3-1.0s`; inserted turn duration is `1-4s`.
- `hold_instead_of_shift`: remove a responder turn of `1.2-8.0s`, then compact
  the resulting hold gap to about `0.5-1.5s`.
- `excessive_backchannel`: insert two short, distinct backchannels by default,
  with `800ms` target spacing and no volume change. To make three-backchannel
  examples, rerun with `--bc-insert-count 3`.

These rules are deliberately not overly restrictive. For cleaner public
datasets, curate the input natural manifest to avoid clips where one channel is
mostly silent, transcripts are missing, or the two speakers are poorly aligned.

Useful generation variants:

```bash
# One perturbation type only
python natural_classification/build_natural_unnatural_dualturn.py make-unnatural \
  --natural-csv data/manifests/test2_natural.csv \
  --out-root data/generated_late \
  --split test2_late \
  --per-type 200 \
  --types late_response \
  --short-context

# Three inserted backchannels instead of the default two
python natural_classification/build_natural_unnatural_dualturn.py make-unnatural \
  --natural-csv data/manifests/test2_natural.csv \
  --out-root data/generated_bc3 \
  --split test2_bc3 \
  --per-type 100 \
  --types excessive_backchannel \
  --bc-insert-count 3 \
  --short-context

# Sharded generation; give each worker a separate output directory
python natural_classification/build_natural_unnatural_dualturn.py make-unnatural \
  --natural-csv data/manifests/test2_natural.csv \
  --out-root data/generated_late_shard00 \
  --split test2 \
  --per-type 200 \
  --types late_response \
  --num-shards 16 \
  --shard-index 0 \
  --short-context
```

## Train FVAD Models

Create `data/manifests/train.csv`, `dev.csv`, and `test1.csv` using the format in
`data/manifests/README.md`. The included configs train with standard FVAD
train/validation loss and frame accuracy. `best.pt` is selected by validation
loss.

For DualTurn all-six training, first cache VAD and signal labels:

```bash
python dualturn/scripts/build_silero_vad_cache.py \
  --manifest data/manifests/train.csv --output-dir data/cache/vad
python dualturn/scripts/build_silero_vad_cache.py \
  --manifest data/manifests/dev.csv --output-dir data/cache/vad --skip-existing
python dualturn/scripts/build_silero_vad_cache.py \
  --manifest data/manifests/test1.csv --output-dir data/cache/vad --skip-existing

python dualturn/scripts/build_dualturn_signal_cache.py \
  --manifest data/manifests/train.csv \
  --manifest data/manifests/dev.csv \
  --manifest data/manifests/test1.csv \
  --vad-cache-dir data/cache/vad --output-dir data/cache/signals
```

Run an experiment:

```bash
python dualturn/scripts/run_fvad_experiment.py configs/train_vap_full.yaml
python dualturn/scripts/run_fvad_experiment.py configs/train_dualturn_native_all6.yaml
python dualturn/scripts/run_fvad_experiment.py configs/train_dualturn_fvad256_all6.yaml
```

Training checkpoints are written to `outputs/<experiment>/checkpoints/` as
`last.pt` and `best.pt`.

## Evaluate Checkpoints

Score a trained FVAD checkpoint on a paired natural/edited manifest:

```bash
python dualturn/scripts/score_fvad_checkpoint.py \
  --checkpoint outputs/dualturn_fvad256_all6/checkpoints/best.pt \
  --experiment auto \
  --manifest data/generated/manifests/test2.csv \
  --output-dir outputs/eval_dualturn_fvad256 \
  --device cuda \
  --batch-size 4
```

For a shared/released checkpoint, download it separately and pass its path with
`--checkpoint`. You can list supported profile names with:

```bash
python dualturn/scripts/score_fvad_checkpoint.py --list-experiments
```

Score an upstream VAP checkpoint directly:

```bash
python dualturn/scripts/score_vap_nll_naturalness.py \
  --unnatural-manifest data/generated/manifests/test2.csv \
  --checkpoint checkpoints/VAP_state_dict.pt \
  --output-dir outputs/eval_vap \
  --device cuda
```

## Metrics

The scorer computes frame-level future-VAD NLL, averages it inside pre-boundary
utterance units, and reports:

| Metric | Definition | Better |
|---|---|---|
| `MeanNLL` | Mean NLL over all utterance units | Lower for a natural recording |
| `TailNLL` | Mean of the worst 25% unit NLL values | Lower |
| `DialogNLL` | `0.5 * MeanNLL + 0.5 * TailNLL` | Lower |
| `DeltaNLL` | `edited DialogNLL - natural DialogNLL` | Positive/larger |
| `Pairwise Accuracy` | Fraction of matched pairs with `DeltaNLL > 0` | Higher |
| `C-index` | Fraction of all edited-vs-natural dialogue-NLL comparisons correctly ordered; ties excluded | Higher |

Results are reported overall and separately for all five edit types. Output is
written under `OUTPUT_DIR/step_<global_step>/`:

- `metrics.json` and `metrics.csv`: aggregate metrics with variance and 95% CI
- `pair_scores.csv`: natural/edited scores and `DeltaNLL` for each pair
- `segment_scores.csv`: recording-level `MeanNLL`, `TailNLL`, and `DialogNLL`
- `units.csv`: utterance-boundary unit NLL values
- `inference_config.json`: resolved experiment profile and checkpoint metadata

These metrics are for standalone evaluation and reporting.

## Samples

`samples/manifest.csv` contains one natural/edited pair for each type:
`early_entry`, `late_response`, `shift_instead_of_hold`,
`hold_instead_of_shift`, and `excessive_backchannel`.

Score the sample pairs with VAP:

```bash
python dualturn/scripts/score_vap_nll_naturalness.py \
  --unnatural-manifest samples/manifest.csv \
  --checkpoint checkpoints/VAP_state_dict.pt \
  --output-dir outputs/sample_vap \
  --device cuda
```

Score the official DualTurn native 8-bit head:

```bash
python dualturn/scripts/score_dualturn_fvad_nll_naturalness.py \
  --unnatural-manifest samples/manifest.csv \
  --output-dir outputs/sample_dualturn \
  --device cuda
```

Large datasets, caches, model weights, and generated outputs are excluded by
`.gitignore`.
