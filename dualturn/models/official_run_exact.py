
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import torch

from dualturn.models.official_wrapper import DualTurnOfficialWrapper



def main() -> None:
    model = DualTurnOfficialWrapper.from_pretrained(
        "anyreach-ai/dualturn-qwen2.5-mimi-0.5B",
        dtype=torch.float32,
    ).eval()

    wav = torch.randn(2, 24000 * 5)
    with torch.no_grad():
        out = model(wav, sr=24000)

    print("vad_probs:", tuple(out.vad_probs.shape))
    print("fvad_probs:", tuple(out.fvad_probs.shape))
    print("eot_probs:", tuple(out.eot_probs.shape))
    print("bot_probs:", tuple(out.bot_probs.shape))
    print("hold_probs:", tuple(out.hold_probs.shape))
    print("bc_probs:", tuple(out.bc_probs.shape))


if __name__ == "__main__":
    main()
