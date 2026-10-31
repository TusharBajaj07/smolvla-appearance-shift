"""Run eval_perturbed.py (official LeRobot LIBERO evaluation + one image perturbation) on Modal.

  .venv-modal/bin/modal run runs/libero/libero_app.py --cond clean --task libero_object --task-ids 0 --n-episodes 5
Results land in the Modal Volume "smolvla-libero" under <task>/<cond>[_patch-<recipe>][_measure]_s<seed>/t<task id>/.
  --patch img_all|img_odd|img_even|text   experiment A (closed-loop KV patching, see eval_perturbed.py)
  --measure                              experiment B (use with --cond clean)
"""
from pathlib import Path

import modal

HERE = Path(__file__).parent
POLICY = "HuggingFaceVLA/smolvla_libero"

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("ffmpeg", "libgl1", "libglib2.0-0", "libegl1", "libgles2", "libosmesa6", "cmake", "build-essential")
    .pip_install("torch==2.10.0", "torchvision==0.25.0", "lerobot[smolvla,libero]==0.4.4")
    .env({"MUJOCO_GL": "egl", "PYOPENGL_PLATFORM": "egl"})
    .add_local_file(HERE / "eval_perturbed.py", "/root/libero/eval_perturbed.py")
    .add_local_file(HERE / "perturb.py", "/root/libero/perturb.py")
)
out_vol = modal.Volume.from_name("smolvla-libero", create_if_missing=True)
hf_vol = modal.Volume.from_name("hf-cache", create_if_missing=True)
app = modal.App("smolvla-libero-eval", image=image)


@app.function(gpu=["L4", "A10G"], cpu=4.0, memory=16384, timeout=6 * 3600, retries=3,  # finished tasks are skipped on retry
              volumes={"/out": out_vol, "/root/.cache/huggingface": hf_vol})
def run_eval(cond: str, task: str, task_ids: str, n_episodes: int, batch: int, seed: int, n_tasks: int = 10,
             patch: str = "", measure: bool = False):
    """Evaluates the tasks one at a time: building all tasks' simulators at once (each with its own EGL
    rendering context) exhausts the GPU's 22 GB before the policy loads."""
    import os
    import subprocess
    import time

    ids = [int(i) for i in task_ids.split(",")] if task_ids else list(range(n_tasks))
    tag = cond + (f"_patch-{patch}" if patch else "") + ("_measure" if measure else "")
    base = f"/out/{task}/{tag}_s{seed}"
    env = dict(os.environ, PERTURB=cond, PERTURB_SEED=str(seed), PATCH=patch)
    t_all = time.time()
    for tid in ids:
        out_dir = f"{base}/t{tid}"
        if os.path.exists(f"{out_dir}/run/eval_info.json"):
            print(f"skip task {tid} (done)", flush=True)
            continue
        os.makedirs(out_dir, exist_ok=True)
        if measure:
            env["MEASURE"] = f"{out_dir}/measure"
        cmd = ["python", "eval_perturbed.py", f"--policy.path={POLICY}", "--env.type=libero", f"--env.task={task}",
               f"--env.task_ids=[{tid}]", f"--eval.n_episodes={n_episodes}", f"--eval.batch_size={batch}",
               "--env.max_parallel_tasks=1", f"--seed={1000 + seed}", f"--output_dir={out_dir}/run"]
        t0 = time.time()
        with open(f"{out_dir}/log.txt", "w") as log:
            # LIBERO asks once whether to use a custom dataset folder; answer "n" (default location)
            proc = subprocess.run(cmd, cwd="/root/libero", env=env, stdout=log, stderr=subprocess.STDOUT,
                                  input="n\n", text=True)
        ok = os.path.exists(f"{out_dir}/run/eval_info.json")
        lines = open(f"{out_dir}/log.txt").read().splitlines()
        for ln in [x for x in lines if "PATCH_CHECK" in x] + ([] if ok else lines[-25:]):
            print(ln, flush=True)
        print(f"TASK {tid} cond={tag} rc={proc.returncode} ok={ok} secs={time.time() - t0:.0f}", flush=True)
        out_vol.commit()
    print(f"EVAL_SECONDS {time.time() - t_all:.0f} cond={tag} task={task} tasks={ids}", flush=True)
    hf_vol.commit()


@app.local_entrypoint()
def main(cond: str = "clean", task: str = "libero_object", task_ids: str = "", n_episodes: int = 10,
         batch: int = 10, seed: int = 0, patch: str = "", measure: bool = False):
    run_eval.remote(cond, task, task_ids, n_episodes, batch, seed, patch=patch, measure=measure)
