import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "learning/12_decode_runtime"))
from decode_runtime import GreedyDecodeGraph


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def test_graph_validates_fixed_inputs():
    with pytest.raises(ValueError, match="shape"):
        GreedyDecodeGraph(None, torch.zeros(2, device="cuda", dtype=torch.long), None, {},
                          torch.zeros(1, device="cuda", dtype=torch.long))

