# Checkpoints

Model weights are intentionally not committed.

## Released TurnNat FVAD Checkpoint

Place the released checkpoint here:

`checkpoints/turnnat-dualturn-fvad256.pt`

Figshare artifact package:

`https://figshare.com/s/65e5bf5290085220ee88`

Expected SHA256:

`96472553406762c662adae1197941d42347b6ada93bb8bccb80479a72b66ba97`

Verify after downloading:

```bash
cd /path/to/turn-taking-naturalness
sha256sum checkpoints/turnnat-dualturn-fvad256.pt
```

This released checkpoint corresponds to the event-weighted DualTurn FVAD-256 run:

- event weight alpha: `8.0`
- minimum utterance duration: `0.2s`

Training configuration/provenance is recorded in
`configs/released_turnnat_dualturn_fvad256/`.

## Other Checkpoints

Place the raw VAP state dict at `VAP-main/example/checkpoints/VAP_state_dict.pt`.
DualTurn base weights are loaded from Hugging Face by default; for offline runs,
pre-cache the DualTurn model and use `--local-files-only`.
