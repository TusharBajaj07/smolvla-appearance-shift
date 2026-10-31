"""Experiments A and B on the LIBERO checkpoint (HuggingFaceVLA/smolvla_libero, LIBERO-Object). CPU only.

A  closed-loop KV patching under strong clutter: success of distractors_2 with clean image (or text) cache entries,
   from modal/distractors_2_patch-<recipe>_s0/t*/run/eval_info.json, compared with clean_s0 and distractors_2_s0
   on the tasks finished by every recipe.
   success recovery = (s_patch - s_pert) / (s_clean - s_pert); 95% CI from a paired bootstrap over tasks.
   A_odd_vs_even: odd- vs even-layer restoration, exact McNemar test over episodes paired by initial state.
B  mechanism during clean rollouts, from modal/clean_measure_s0/t*/measure/batch*.npz (valid frames only):
   drift of the mean image-token residual per VLM layer in rulers (median clean distance between frames of different
   episodes), action change |a_pert - a_clean|, and offline patching recovery 1 - |a_r - a_c| / |a_none - a_c|
   (ratio of means; 95% CI from a bootstrap over episodes), within-type and pooled Spearman of drift vs change.

  .venv-lerobot/bin/python -I runs/libero/analyze_patch.py   ->  runs/libero/results_patch.json
"""
import json
from math import comb
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
M = HERE / "modal"
STRONG = ["dark_2", "blur_2", "distractors_2", "camera_shift_2"]
RECIPES = ["img_all", "img_odd", "img_even", "text"]
N_BOOT, rng = 2000, np.random.default_rng(0)


def success(tag):
    out = {}
    for f in sorted((M / tag).glob("t*/run/eval_info.json")):
        for t in json.loads(f.read_text())["per_task"]:
            out[t["task_id"]] = float(np.mean(t["metrics"]["successes"]))
    return out


def rank(x):
    return np.argsort(np.argsort(x)).astype(float)


def spearman(a, b):
    return float(np.corrcoef(rank(a), rank(b))[0, 1])


def boot_groups(groups, stat, n=N_BOOT):
    """95% CI of stat(mask-free index array) resampling whole groups (episodes)."""
    ug = np.unique(groups)
    pos = {g: np.where(groups == g)[0] for g in ug}
    vals = []
    for _ in range(n):
        idx = np.concatenate([pos[g] for g in rng.choice(ug, len(ug))])
        vals.append(stat(idx))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def exp_a():
    clean, pert = success("clean_s0"), success("distractors_2_s0")
    runs = {r: success(f"distractors_2_patch-{r}_s0") for r in RECIPES}
    runs = {r: p for r, p in runs.items() if p}
    # every recipe is compared on the same tasks: those finished by all recipes (runs were stopped early)
    tasks = sorted(set(clean) & set(pert) & set.intersection(*(set(p) for p in runs.values()))) if runs else []
    res = {}
    for r, p in runs.items():
        c, q, s = (np.array([d[t] for t in tasks]) for d in (clean, pert, p))
        b = rng.integers(0, len(tasks), (N_BOOT, len(tasks)))
        rec = lambda i: (s[i].mean() - q[i].mean()) / max(c[i].mean() - q[i].mean(), 1e-9)  # noqa: E731
        res[r] = {"n_tasks": len(tasks), "success": float(s.mean()), "clean": float(c.mean()),
                  "perturbed": float(q.mean()), "success_recovery": float(rec(np.arange(len(tasks)))),
                  "success_recovery_ci": [float(np.percentile([rec(i) for i in b], x)) for x in (2.5, 97.5)],
                  "success_ci": [float(np.percentile(s[b].mean(1), x)) for x in (2.5, 97.5)]}
        print(f"A {r:9s} success {s.mean() * 100:5.1f}% (clean {c.mean() * 100:.1f}, clutter {q.mean() * 100:.1f}) "
              f"recovery {res[r]['success_recovery']:.2f} {res[r]['success_recovery_ci']}  [{len(tasks)} tasks]")
    return res


def episode_successes(tag):
    """Per task: success of each episode (episode i has the same initial state and clutter layout in every run)."""
    out = {}
    for f in sorted((M / tag).glob("t*/run/eval_info.json")):
        for t in json.loads(f.read_text())["per_task"]:
            out[t["task_id"]] = np.array(t["metrics"]["successes"], float)
    return out


def exp_a_odd_vs_even():
    """Paired comparison of odd-layer vs even-layer image restoration on the common tasks.
    Exact McNemar test on episodes (paired by initial state) and a paired bootstrap over tasks of the success
    difference. Uses its own generator so the other results stay bit-identical to earlier runs."""
    runs = {r: episode_successes(f"distractors_2_patch-{r}_s0") for r in RECIPES}
    if not all(runs.values()):
        return None
    clean, pert = success("clean_s0"), success("distractors_2_s0")
    tasks = sorted(set(clean) & set(pert) & set.intersection(*(set(p) for p in runs.values())))
    o = np.concatenate([runs["img_odd"][t] for t in tasks])
    e = np.concatenate([runs["img_even"][t] for t in tasks])
    b, c = int(((o == 1) & (e == 0)).sum()), int(((o == 0) & (e == 1)).sum())
    n = b + c
    p = min(1.0, 2 * sum(comb(n, k) for k in range(min(b, c) + 1)) / 2 ** n) if n else 1.0
    O = np.array([runs["img_odd"][t].mean() for t in tasks])
    E = np.array([runs["img_even"][t].mean() for t in tasks])
    idx = np.random.default_rng(1).integers(0, len(tasks), (10000, len(tasks)))
    d = (O[idx] - E[idx]).mean(1)
    res = {"n_tasks": len(tasks), "n_episodes": int(len(o)), "odd_only": b, "even_only": c, "mcnemar_p": float(p),
           "diff": float((O - E).mean()), "diff_ci": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]}
    print(f"A odd vs even: {int(o.sum())}/{len(o)} vs {int(e.sum())}/{len(e)}; discordant {b} vs {c}, "
          f"exact McNemar p = {p:.3f}; diff {res['diff']:.2f} {res['diff_ci']}")
    return res


def load_b():
    recs = []
    for d in sorted((M / "clean_measure_s0").glob("t*/measure")):
        tid = int(d.parent.name[1:])
        for f in sorted(d.glob("batch*_step*.npz")):  # one file per measurement, arrays [n_envs, ...]
            z = dict(np.load(f))
            z["episode"] = tid * 1000 + int(f.stem.split("_")[0][5:]) * 100 + z["env"]
            recs.append(z)
    if not recs:
        return None
    z = {k: np.concatenate([r[k] for r in recs]) for k in recs[0]}
    keep = z["valid"].astype(bool)
    return {k: v[keep] for k, v in z.items()}


def ruler(F, ep):
    """Per layer: median distance between clean features of frames from different episodes."""
    i, j = rng.integers(0, len(F), (2, 20000))
    m = ep[i] != ep[j]
    return np.median(np.linalg.norm(F[i[m]] - F[j[m]], axis=-1), 0)


def exp_b():
    z = load_b()
    if z is None:
        return None
    ep = z["episode"]
    Fc = z["feat_img_clean"].astype(np.float32)  # [N, 17, 960]
    R = ruler(Fc, ep)
    res = {"n_frames": int(len(ep)), "n_episodes": int(len(np.unique(ep))), "conditions": {}}
    n_layers = Fc.shape[1] - 1  # this checkpoint uses all 32 SmolVLM2 layers (the SO-101 one used 16)
    res["n_layers"] = int(n_layers)
    recipes = ["img_all", "img_odd", "img_even", "text"] + [f"single_{L}" for L in range(n_layers)]
    d_all, e_all = [], []
    for c in STRONG:
        drift = np.linalg.norm(z[f"feat_img_{c}"].astype(np.float32) - Fc, axis=-1) / R  # [N, 17]
        base = np.abs(z[f"act_{c}_none"] - z["act_clean"]).mean((1, 2))
        r = {"drift_by_layer": drift.mean(0).tolist(), "action_change": float(base.mean()),
             "within_spearman": spearman(drift[:, 0], base), "recovery": {}}
        for k in recipes:
            per = np.abs(z[f"act_{c}_{k}"] - z["act_clean"]).mean((1, 2))
            stat = lambda i: 1 - per[i].mean() / base[i].mean()  # noqa: E731
            r["recovery"][k] = [float(stat(np.arange(len(per)))), boot_groups(ep, stat, 500)]
        r["mean_single_odd"] = float(np.mean([r["recovery"][f"single_{L}"][0] for L in range(1, n_layers, 2)]))
        r["mean_single_even"] = float(np.mean([r["recovery"][f"single_{L}"][0] for L in range(0, n_layers, 2)]))
        res["conditions"][c] = r
        d_all.append(drift[:, 0])
        e_all.append(base)
        rr = r["recovery"]
        print(f"B {c:15s} drift L0 {r['drift_by_layer'][0]:.2f} mid {r['drift_by_layer'][n_layers // 2]:.2f} "
              f"last {r['drift_by_layer'][-1]:.2f} | change {r['action_change']:.3f} | rec img {rr['img_all'][0]:.2f} "
              f"odd {rr['img_odd'][0]:.2f} even {rr['img_even'][0]:.2f} text {rr['text'][0]:.2f} | single odd "
              f"{r['mean_single_odd']:.2f} even {r['mean_single_even']:.2f} | within rho {r['within_spearman']:.2f}")
    res["pooled_spearman"] = spearman(np.concatenate(d_all), np.concatenate(e_all))
    res["mean_within_spearman"] = float(np.mean([v["within_spearman"] for v in res["conditions"].values()]))
    print(f"B {res['n_frames']} frames, {res['n_episodes']} episodes | pooled rho {res['pooled_spearman']:.2f}, "
          f"mean within {res['mean_within_spearman']:.2f}")
    return res


if __name__ == "__main__":
    out = {"A": exp_a(), "A_odd_vs_even": exp_a_odd_vs_even(), "B": exp_b()}
    (HERE / "results_patch.json").write_text(json.dumps(out, indent=1))
    print("saved", HERE / "results_patch.json")
