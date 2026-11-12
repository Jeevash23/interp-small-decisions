"""D3 round 2, question 5: download OLMo-2-0425-1B training checkpoints (about 5.9 GB each) into the Hugging Face
cache, one at a time, stopping if fewer than 50 GB would remain free (owner's rule, 8 Nov 17:00).
Usage: python r2_dl_olmo_ckpts.py rev [rev ...]"""
import shutil, sys, time
from huggingface_hub import snapshot_download
for rev in sys.argv[1:]:
    free = shutil.disk_usage("/").free / 1e9
    if free - 6.0 < 50: print(f"stop: {free:.1f} GB free", flush=True); break
    t0 = time.time()
    p = snapshot_download("allenai/OLMo-2-0425-1B", revision=rev, allow_patterns=["*.safetensors", "*.json", "*.txt"])
    print(f"{rev} done in {time.time() - t0:.0f} s; {shutil.disk_usage('/').free / 1e9:.1f} GB free; {p}", flush=True)
