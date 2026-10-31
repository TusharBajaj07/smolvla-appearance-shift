"""Second-round analyses (CPU): seeds, action-expert drift, activation patching. Writes results_more.json.

SEEDS     Seeds 1 and 2 change distractor placement and the starting noise. Per seed: mean action change per
          condition, pooled and within-condition Spearman(drift, action change); distractor drift per seed.
EXPERT    Action expert residual stream (17 points x denoising steps 0/3/6/9, mean over the 50 action tokens):
          drift per layer/step in rulers (median distance between clean frames of different episodes), and
          within-condition Spearman between expert drift and action change (where do per-frame action changes
          become predictable?).
PATCH     Recovery = 1 - |a_patched - a_clean| / |a_perturbed - a_clean| (mean over frames, 50 steps, 6 joints),
          per recipe and condition, 95% episode-bootstrap CI. 1 = patch fully restores the clean action plan.

  .venv-lerobot/bin/python runs/drift_study/analyze_more.py
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.argv = sys.argv[:1]
sys.path.insert(0, str(HERE))
import analyze as A  # noqa: E402  (boot_ci, ruler, spearman, CONDS)

M = HERE / "modal"
STRONG = ["dark_2", "blur_2", "distractors_2", "camera_shift_2"]


def load_actions(seed):
    d = M / ("actions_sub300" + (f"_s{seed}" if seed else ""))
    a0 = np.load(d / "clean.npz")["actions"]
    return np.load(d / "meta.npz"), {c: np.abs(np.load(d / f"{c}.npz")["actions"] - a0).mean((1, 2)) for c in A.CONDS}


def seeds():
    fmeta = np.load(M / "features_all/meta.npz")
    pos = {int(i): k for k, i in enumerate(fmeta["index"])}
    H0 = np.load(M / "features_all/clean.npz")["image"].astype(np.float32)
    R = A.ruler(H0, fmeta["episode"])
    out = {}
    for seed in (0, 1, 2):
        am, err = load_actions(seed)
        rows = np.array([pos[int(i)] for i in am["index"]])
        drift, within, d_rows = {}, {}, []
        for c in A.CONDS:
            # only distractor placement depends on the seed; other perturbations are deterministic
            fdir = M / ("features_all" + (f"_s{seed}" if seed and c.startswith("distractors") else ""))
            H = np.load(fdir / f"{c}.npz")["image"][:, 0].astype(np.float32)
            d_all = np.linalg.norm(H - H0[:, 0], axis=-1) / R[0]
            drift[c] = float(d_all.mean())
            within[c] = A.spearman(d_all[rows], err[c])
            d_rows.append(d_all[rows])
        d_cat, e_cat = np.concatenate(d_rows), np.concatenate([err[c] for c in A.CONDS])
        out[seed] = {"action_change": {c: float(err[c].mean()) for c in A.CONDS}, "drift_L0": drift,
                     "within_spearman": within, "pooled_spearman": A.spearman(d_cat, e_cat)}
        print(f"seed {seed}: pooled rho {out[seed]['pooled_spearman']:.3f} | mean within {np.mean(list(within.values())):.3f} | "
              + " ".join(f"{c} {err[c].mean():.2f}" for c in A.CONDS), flush=True)
    summary = {c: {"action_mean": float(np.mean([out[s]["action_change"][c] for s in out])),
                   "action_sd_over_seeds": float(np.std([out[s]["action_change"][c] for s in out], ddof=1))}
               for c in A.CONDS}
    return {"per_seed": out, "summary": summary}


def expert():
    d = M / "expert_sub300"
    meta = np.load(d / "meta.npz")
    ep = meta["episode"]
    _, err = load_actions(0)
    E0 = np.load(d / "clean.npz")["expert"].astype(np.float32)  # [N, 4 steps, 17, 720]
    n_steps, n_pts = E0.shape[1], E0.shape[2]
    R = np.stack([A.ruler(E0[:, s], ep) for s in range(n_steps)])  # [4, 17]
    out = {"steps": [0, 3, 6, 9], "conditions": {}}
    for c in A.CONDS:
        E = np.load(d / f"{c}.npz")["expert"].astype(np.float32)
        drift = np.linalg.norm(E - E0, axis=-1) / R[None]  # [N, 4, 17]
        rho = [[A.spearman(drift[:, s, L], err[c]) for L in range(n_pts)] for s in range(n_steps)]
        out["conditions"][c] = {"drift": drift.mean(0).tolist(), "within_spearman": rho}
    mean_rho = np.mean([out["conditions"][c]["within_spearman"] for c in A.CONDS], axis=0)  # [4, 17]
    out["mean_within_spearman"] = mean_rho.tolist()
    print("expert: mean within-condition rho(expert drift, action change), step 0 / step 9, layers 0,4,8,12,16:")
    for s in (0, 3):
        print(f"  step {out['steps'][s]}: " + " ".join(f"L{L} {mean_rho[s, L]:.2f}" for L in (0, 4, 8, 12, 16)), flush=True)
    for c in STRONG:
        dr = np.array(out["conditions"][c]["drift"])
        print(f"  {c:15s} expert drift step0 L1 {dr[0, 1]:.2f} L16 {dr[0, 16]:.2f} | step9 L1 {dr[3, 1]:.2f} L16 {dr[3, 16]:.2f}")
    return out


def patch():
    d = M / "patch_sub300"
    ep = np.load(d / "meta.npz")["episode"]
    out = {}
    for c in STRONG:
        p = np.load(d / f"{c}.npz")
        clean = p["all"]
        base = np.abs(p["none"] - clean).mean((1, 2))  # per frame

        def recovery(name):
            per = np.abs(p[name] - clean).mean((1, 2))
            stat = lambda m: 1 - m[:, 0].mean() / m[:, 1].mean()  # noqa: E731
            both = np.stack([per, base], 1)
            return float(stat(both)), A.boot_ci(both, ep, stat=stat)

        rec = {name: recovery(name) for name in p.files if name not in ("none", "all")}
        out[c] = {"base_change": float(base.mean()), "recovery": {k: v[0] for k, v in rec.items()},
                  "recovery_ci": {k: v[1] for k, v in rec.items()}}
        r = out[c]["recovery"]
        print(f"patch {c:15s} base {base.mean():.2f} | tok_img {r['tok_img']:.2f} tok_text {r['tok_text']:.2f} "
              f"tok_up {r['tok_up']:.2f} tok_side {r['tok_side']:.2f} | single L0 {r['single_0']:.2f} L7 {r['single_7']:.2f} "
              f"L15 {r['single_15']:.2f} | upto L3 {r['upto_3']:.2f} L7 {r['upto_7']:.2f} L11 {r['upto_11']:.2f}", flush=True)
    return out


if __name__ == "__main__":
    res = {}
    for name, fn in (("seeds", seeds), ("expert", expert), ("patch", patch)):
        try:
            res[name] = fn()
        except FileNotFoundError as e:
            print(f"{name}: skipped ({e})")
    (HERE / "results_more.json").write_text(json.dumps(res, indent=1))
    print("saved", HERE / "results_more.json")
