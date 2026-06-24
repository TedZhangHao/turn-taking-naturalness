# Checkpoints

Model weights are intentionally not committed. Place the raw VAP state dict at
`checkpoints/VAP_state_dict.pt`. DualTurn base weights are loaded from Hugging
Face; trained checkpoints are written under `outputs/<run>/checkpoints/`.
