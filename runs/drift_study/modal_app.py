"""Run study.py on a Modal GPU. The model code runs remotely; this file only describes the machine and launches it.

  .venv-modal/bin/modal run runs/drift_study/modal_app.py --mode features --frames demo32
  .venv-modal/bin/modal volume get smolvla-drift features_demo32 runs/drift_study/modal/

Machine: Debian + Python 3.10, torch/torchvision/lerobot pinned to the Mac's versions so results are comparable,
one NVIDIA L4 (24 GB). Outputs go to the Modal Volume "smolvla-drift"; the Hugging Face cache (model + dataset,
~1 GB) is kept in the Volume "hf-cache" so it downloads only once.
"""
from pathlib import Path

import modal

HERE = Path(__file__).parent
GPU = "L4"

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("ffmpeg", "libgl1", "libglib2.0-0")
    .pip_install("torch==2.10.0", "torchvision==0.25.0", "lerobot[smolvla]==0.4.4")
    .add_local_file(HERE / "study.py", "/root/drift_study/study.py")
    .add_local_file(HERE / "perturb.py", "/root/drift_study/perturb.py")
)
out_vol = modal.Volume.from_name("smolvla-drift", create_if_missing=True)
hf_vol = modal.Volume.from_name("hf-cache", create_if_missing=True)
ckpt_vol = modal.Volume.from_name("smolvla-heldout", create_if_missing=True)  # held-out fine-tunes
app = modal.App("smolvla-drift-study", image=image)


@app.function(gpu=GPU, volumes={"/out": out_vol, "/root/.cache/huggingface": hf_vol, "/ckpt": ckpt_vol}, timeout=4 * 3600)
def run(args: list):
    import subprocess
    import sys

    print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout, flush=True)
    sys.path.insert(0, "/root/drift_study")
    import study

    try:
        study.main(args + ["--out", "/out"])
    finally:  # keep whatever finished, even on failure
        out_vol.commit()
        hf_vol.commit()


@app.local_entrypoint()
def main(mode: str, frames: str, conditions: str = "all", limit: int = 0, batch: int = 16, seed: int = 0,
         model: str = "", episodes: str = "", n_frames: int = 0, tag: str = ""):
    args = ["--mode", mode, "--frames", frames, "--conditions", conditions, "--batch", str(batch), "--seed", str(seed)]
    if model:
        args += ["--model", model]
    if episodes:
        args += ["--episodes", episodes]
    if n_frames:
        args += ["--n-frames", str(n_frames)]
    if tag:
        args += ["--tag", tag]
    if limit:
        args += ["--limit", str(limit)]
    run.remote(args)
