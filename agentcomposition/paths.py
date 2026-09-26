"""Portable paths; model weights and experiment inputs are supplied separately."""
import os
from pathlib import Path
ROOT = Path(os.environ.get("AGENTCOMPOSITION_ROOT", Path(__file__).resolve().parents[1])).resolve()
BASE_MODEL = Path(os.environ.get("BASE_MODEL", ROOT / "checkpoints/base/Meta-Llama-3-8B"))
EASYREC_MODEL = Path(os.environ.get("EASYREC_MODEL", ROOT / "checkpoints/base/EasyRec_tuned"))
def env(name, default):
    return os.environ.get(name, default)
