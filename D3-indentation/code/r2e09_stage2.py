"""D3 round 2, r2e09 (exploratory, question 5): run r2e08_ckpt.py on OLMo-2-0425-1B stage-2 checkpoints. r2common.py is
frozen with PROTOCOL-2 and registers stage-1 checkpoints only, so this wrapper registers stage-2 checkpoints found in the
Hugging Face cache as models "olmo1b-s2i<ingredient>-<tokens>" and then runs r2e08 unchanged.
Usage: python r2e09_stage2.py <n_chunks> <model> [<model> ...]"""
import sys, runpy
from pathlib import Path
import torch
import r2common
hub = Path.home() / ".cache" / "huggingface" / "hub" / "models--allenai--OLMo-2-0425-1B"
for ref in (hub / "refs").glob("stage2-*"):
    _, ing, _, tok = ref.name.split("-")
    r2common.MODELS[f"olmo1b-s2{ing.replace('ingredient', 'i')}-{tok.replace('tokens', '')}"] = (str(hub / "snapshots" / ref.read_text().strip()), torch.bfloat16)
sys.argv = ["r2e08_ckpt.py"] + sys.argv[1:]
runpy.run_path(str(Path(__file__).parent / "r2e08_ckpt.py"), run_name="__main__")
