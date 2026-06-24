# Turn-Taking Naturalness

Minimal code for generating five conversational timing edits, training VAP or
DualTurn FVAD models, and scoring paired natural/edited conversations with NLL.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e VAP-main
```

Place a raw VAP model state dict from the
[upstream VAP repository](https://github.com/ErikEkstedt/VAP) at
`checkpoints/VAP_state_dict.pt`. DualTurn is loaded from
`anyreach-ai/dualturn-qwen2.5-mimi-0.5B` through Hugging Face.

## Inference

The default inference profile is `group4-dualturn-full-all6-fvad256`. Put the
shared checkpoint at:

```text
checkpoints/group4-dualturn-full-all6-fvad256.pt
```

Then score the five included sample pairs:

```bash
cd checkpoints && sha256sum -c group4-dualturn-full-all6-fvad256.pt.sha256 && cd ..
python dualturn/scripts/score_fvad_checkpoint.py --device cuda
```

For another checkpoint or dataset:

```bash
python dualturn/scripts/score_fvad_checkpoint.py \
  --experiment group4-dualturn-full-all6-fvad256 \
  --checkpoint /path/to/checkpoint.pt \
  --manifest /path/to/test2.csv \
  --output-dir outputs/my_test \
  --device cuda
```

The scorer validates the selected profile against metadata when available.
Compact mid-training checkpoints may omit metadata, so keep an explicit
`--experiment`; `--experiment auto` requires a full `last.pt` or `best.pt`.
List available profiles with:

```bash
python dualturn/scripts/score_fvad_checkpoint.py --list-experiments
```

Supported profiles include Group 3 VAP/DualTurn heads and Group 4 VAP full,
DualTurn LoRA/full, native FVAD, all-six, and FVAD-256 experiments. Checkpoints
must be produced by `train_fvad_head.py` so they contain model and experiment
metadata. The default FVAD-256 checkpoint uses full 256-state categorical NLL;
native DualTurn profiles use summed 8-bit Bernoulli joint NLL.

Model checkpoints are large and ignored by Git. Share them with Hugging Face,
Git LFS, or external storage rather than a normal GitHub commit.

Before sharing a training checkpoint, remove optimizer state and the lazily
cached Mimi copy while adding complete experiment metadata:

```bash
python dualturn/scripts/export_inference_checkpoint.py \
  --input /path/to/best_naturalness_c_index.pt \
  --output checkpoints/group4-dualturn-full-all6-fvad256.pt \
  --experiment group4-dualturn-full-all6-fvad256
```

Send the exported file together with this repository. The recipient can then
run the default one-line inference command above.

## Metrics

The scorer first computes frame NLL, averages it inside 2-second pre-boundary
utterance units, and reports:

| Metric | Definition | Better |
|---|---|---|
| `MeanNLL` | Mean NLL over all utterance units | Lower for a natural recording |
| `TailNLL` | Mean of the worst 25% unit NLL values | Lower |
| `DialogNLL` | `0.5 * MeanNLL + 0.5 * TailNLL` | Lower |
| `DeltaNLL` | `edited DialogNLL - natural DialogNLL` | Positive/larger |
| `Pairwise Accuracy` | Fraction of matched pairs with `DeltaNLL > 0` | Higher |
| `C-index` | Fraction of all edited-vs-natural comparisons correctly ordered; ties excluded | Higher |

Results are reported overall and separately for all five edit types. Output is
written under `OUTPUT_DIR/step_<global_step>/`:

- `metrics.json` and `metrics.csv`: aggregate metrics with variance and 95% CI.
- `pair_scores.csv`: natural/edited scores and DeltaNLL for each pair.
- `segment_scores.csv`: recording-level Mean/Tail/Dialog NLL.
- `units.csv`: utterance-boundary unit NLL values.
- `inference_config.json`: resolved experiment profile and checkpoint metadata.

## Samples

`samples/manifest.csv` contains one natural/edited pair for each type:
`early_entry`, `late_response`, `shift_instead_of_hold`,
`hold_instead_of_shift`, and `excessive_backchannel`.

Score the five sample pairs with VAP:

```bash
python dualturn/scripts/score_vap_nll_naturalness.py \
  --unnatural-manifest samples/manifest.csv \
  --checkpoint checkpoints/VAP_state_dict.pt \
  --output-dir outputs/sample_vap \
  --device cuda
```

Score the official DualTurn native 8-bit head (summed joint NLL):

```bash
python dualturn/scripts/score_dualturn_fvad_nll_naturalness.py \
  --unnatural-manifest samples/manifest.csv \
  --output-dir outputs/sample_dualturn \
  --device cuda
```

## Train

Create `data/manifests/train.csv`, `dev.csv`, and `test1.csv` using the format in
`data/manifests/README.md`. For DualTurn all-six training, first cache VAD and
signal labels:

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

Generate edited data from a natural manifest:

```bash
python natural_classification/build_natural_unnatural_dualturn.py make-unnatural \
  --natural-csv data/manifests/test2_natural.csv \
  --out-root data/generated --per-type 100 \
  --types early_entry,late_response,shift_instead_of_hold,hold_instead_of_shift,excessive_backchannel \
  --turn-source silero --short-context
```

Large datasets, caches, model weights, and outputs are excluded by `.gitignore`.
