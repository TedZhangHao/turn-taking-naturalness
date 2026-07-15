# Released TurnNat DualTurn FVAD-256 Configuration

This directory records the configuration provenance for the released
`turnnat-dualturn-fvad256.pt` checkpoint.

Files:

- `training_overrides.yaml`: static experiment override YAML used for the
  DualTurn full fine-tuning FVAD-256/all-loss profile.
- `base_otospeech_dualturn.sanitized.yaml`: sanitized base YAML referenced by
  the override file. Local absolute paths are replaced by placeholders.
- `training_config.sanitized.json`: sanitized training-time configuration
  snapshot saved with the checkpoint output artifacts. This is the most useful
  file for run-specific settings such as event weighting.

Important run-specific settings:

- checkpoint file: `checkpoints/turnnat-dualturn-fvad256.pt`
- original checkpoint step: `57000`
- FVAD head: `categorical256`
- target scheme: `shared-binary`
- losses: `all`
- event weight alpha: `8.0`
- minimum utterance duration: `0.2s`
- unit window: `pre=2.0s`, `post=0.0s`
- unit mode: `boundaries`
- benchmark manifest used during training/evaluation:
  `dataset/turn_taking_naturalness_5types/manifests/test2.csv`

The public benchmark uses neutralized paths but the same 1,000 pair IDs and the
same underlying audio as the original internal benchmark manifest.
