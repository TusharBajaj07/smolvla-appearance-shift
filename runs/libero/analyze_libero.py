"""Closed-loop LIBERO success under image perturbations (HuggingFaceVLA/smolvla_libero, LIBERO-Object).

Reads runs/libero/modal/<cond>/eval_info.json (downloaded from the "smolvla-libero" volume) and writes
runs/libero/results_libero.json with, per condition: overall success, per-task success, and a 95% CI from a
bootstrap over tasks (episodes within a task share the task's difficulty). Differences to clean are paired:
every condition runs the same tasks and episode seeds, so the bootstrap resamples tasks jointly.

  .venv-lerobot/bin/python runs/libero/analyze_libero.py
"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
CONDS = ["clean", "dark_2", "blur_2", "distractors_2", "camera_shift_2"]
N_BOOT, rng = 2000, np.random.default_rng(0)


def load(cond):
    """Tasks were evaluated one at a time: modal/<cond>_s0/t<i>/run/eval_info.json."""
    out = {}
    for f in sorted((HERE / "modal" / f"{cond}_s0").glob("t*/run/eval_info.json")):
        for t in json.loads(f.read_text())["per_task"]:
            out[t["task_id"]] = np.array(t["metrics"]["successes"], dtype=float)
    return out


def main():
    data = {c: d for c in CONDS if (d := load(c))}
    common = sorted(set.intersection(*(set(d) for d in data.values())))
    data = {c: {t: d[t] for t in common} for c, d in data.items()}
    task_ids = sorted(data["clean"])
    per_task = {c: np.array([data[c][t].mean() for t in task_ids]) for c in data}
    out = {"suite": "libero_object", "task_ids": task_ids,
           "episodes_per_task": int(len(data["clean"][task_ids[0]])), "conditions": {}}
    boots = rng.integers(0, len(task_ids), (N_BOOT, len(task_ids)))
    for c in data:
        s = per_task[c]
        res = {"success": float(s.mean()), "per_task": s.tolist(),
               "ci": [float(np.percentile(s[boots].mean(1), q)) for q in (2.5, 97.5)]}
        if c != "clean":
            diff = s - per_task["clean"]
            res["drop_vs_clean"] = float(-diff.mean())
            res["drop_ci"] = [float(np.percentile(-diff[boots].mean(1), q)) for q in (2.5, 97.5)]
        out["conditions"][c] = res
        extra = f" | drop {res['drop_vs_clean'] * 100:.0f} pts [{res['drop_ci'][0] * 100:.0f}, {res['drop_ci'][1] * 100:.0f}]" if c != "clean" else ""
        print(f"{c:15s} success {res['success'] * 100:5.1f}% [{res['ci'][0] * 100:.0f}, {res['ci'][1] * 100:.0f}]{extra}")
    (HERE / "results_libero.json").write_text(json.dumps(out, indent=1))
    print("saved", HERE / "results_libero.json")


if __name__ == "__main__":
    main()
