"""Held-out check, part 1: fine-tune SmolVLA on 40 of the 50 episodes (10 held out), on Modal.

Same recipe as the public checkpoint (dajiangw/smolvla_svla_so101_pickplace_finetune, see its train_config.json):
start from lerobot/smolvla_base, batch 64, AdamW lr 1e-4, cosine decay to 2.5e-6 by step 30k, seed 1000,
frozen VLM (train_expert_only), no image augmentation. Only change: 30k steps instead of 100k (the learning rate
is already at its floor after 30k) and the 10 held-out episodes in split.json are excluded.

  .venv-modal/bin/modal run runs/heldout/train_app.py --steps 300      # timing test
  .venv-modal/bin/modal run runs/heldout/train_app.py --steps 30000    # full run
"""
import json
from pathlib import Path

import modal

HERE = Path(__file__).parent  # split.json is read on the Mac in main() and passed in; the container never reads it

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("ffmpeg", "libgl1", "libglib2.0-0")
    .pip_install("torch==2.10.0", "torchvision==0.25.0", "lerobot[smolvla]==0.4.4")
)
out_vol = modal.Volume.from_name("smolvla-heldout", create_if_missing=True)
hf_vol = modal.Volume.from_name("hf-cache", create_if_missing=True)
app = modal.App("smolvla-heldout-train", image=image)


@app.function(gpu="L4", cpu=8.0, memory=32768, timeout=6 * 3600,
              volumes={"/out": out_vol, "/root/.cache/huggingface": hf_vol})
def train(steps: int, run_name: str, train_episodes: list, amp: bool = False):
    import subprocess
    import time

    out_dir = f"/out/{run_name}"
    cmd = ["lerobot-train",
           "--policy.path=lerobot/smolvla_base",
           "--dataset.repo_id=lerobot/svla_so101_pickplace",
           f"--dataset.episodes={json.dumps(train_episodes)}",
           "--batch_size=64", f"--steps={steps}", "--seed=1000", "--num_workers=6",
           f"--save_freq={max(steps // 3, 100)}", "--log_freq=100", "--eval_freq=0",
           f"--output_dir={out_dir}", f"--job_name={run_name}",
           "--policy.device=cuda", "--policy.push_to_hub=false", "--wandb.enable=false",
           f"--policy.use_amp={'true' if amp else 'false'}",  # mixed-precision training (speed); analysis stays fp32
           # same camera mapping as the public fine-tune's train_config.json (camera3 stays empty)
           '--rename_map={"observation.images.up": "observation.images.camera1", '
           '"observation.images.side": "observation.images.camera2"}']
    print(" ".join(cmd), flush=True)
    t0 = time.time()
    try:
        subprocess.run(cmd, check=True)
    finally:
        print(f"TRAIN_SECONDS {time.time() - t0:.0f} for {steps} steps", flush=True)
        out_vol.commit()
        hf_vol.commit()


@app.local_entrypoint()
def main(steps: int = 300, run_name: str = "", gpu: str = "L4", amp: bool = False):
    split = json.loads((HERE / "split.json").read_text())
    assert len(split["train"]) == 40 and not set(split["train"]) & set(split["heldout"])
    name = run_name or f"smolvla_40ep_{steps}steps_{gpu}{'_amp' if amp else ''}"
    train.with_options(gpu=gpu).remote(steps, name, split["train"], amp)
