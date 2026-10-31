"""Held-out check, part 2: compare the 40-episode model on SEEN vs UNSEEN (held-out) episodes. CPU only.

Inputs (downloaded from the Modal volume "smolvla-drift" into runs/heldout/modal/):
  actions_custom_{heldout,seen}   new model, clean + 8 perturbations, shared noise
  expert_custom_{heldout,seen}    new model, expert residual stream at denoising steps 0/3/6/9
  patch_custom_{heldout,seen}     new model, KV-cache patching, 4 strong perturbations
  actions_custom_origheldout      original public model, clean only, same held-out frames (quality reference)
VLM features come from runs/drift_study/modal/features_all (the VLM is frozen and identical in every model).

Writes runs/heldout/results_heldout.json.
  .venv-lerobot/bin/python runs/heldout/analyze_heldout.py
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
ROOT = HERE.parents[1]
sys.argv = sys.argv[:1]
sys.path.insert(0, str(ROOT / "runs/drift_study"))
import analyze as A  # noqa: E402  (CONDS, boot_ci, ruler, spearman)

M = HERE / "modal"
FEAT = ROOT / "runs/drift_study/modal/features_all"
STRONG = ["dark_2", "blur_2", "distractors_2", "camera_shift_2"]
CHUNK = 50


def recorded_future(idx):
    """Recorded actions for the next 50 steps after each frame (repeat the last action past the episode end)."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    ds = LeRobotDataset("lerobot/svla_so101_pickplace")
    act = np.array(ds.hf_dataset["action"], dtype=np.float32)
    ep = np.array(ds.hf_dataset["episode_index"])
    out = []
    for i in idx:
        j = np.arange(i, i + CHUNK)
        j = np.minimum(j, np.where(ep == ep[i])[0].max())
        out.append(act[j])
    return np.stack(out)


def one_split(name, feat_meta, H0, R0):
    d = M / f"actions_custom_{name}"
    am = np.load(d / "meta.npz")
    ep = am["episode"]
    pos = {int(i): k for k, i in enumerate(feat_meta["index"])}
    rows = np.array([pos[int(i)] for i in am["index"]])
    A0 = np.load(d / "clean.npz")["actions"]
    res = {"n_frames": int(len(ep)), "episodes": sorted(set(int(e) for e in ep)), "conditions": {}}

    rec = recorded_future(am["index"])
    res["clean_mae_vs_recorded"] = float(np.abs(A0 - rec).mean())
    res["clean_mae_vs_recorded_ci"] = A.boot_ci(np.abs(A0 - rec).mean((1, 2)), ep)

    d_cat, e_cat = [], []
    for c in A.CONDS:
        e = np.abs(np.load(d / f"{c}.npz")["actions"] - A0).mean((1, 2))
        H = np.load(FEAT / f"{c}.npz")["image"][rows, 0].astype(np.float32)
        drift = np.linalg.norm(H - H0[rows], axis=-1) / R0
        both = np.stack([drift, e], 1)
        res["conditions"][c] = {
            "action_change": float(e.mean()), "action_change_ci": A.boot_ci(e, ep),
            "within_spearman": A.spearman(drift, e),
            "within_spearman_ci": A.boot_ci(both, ep, stat=lambda m: A.spearman(m[:, 0], m[:, 1])),
        }
        d_cat.append(drift)
        e_cat.append(e)
    res["pooled_spearman"] = A.spearman(np.concatenate(d_cat), np.concatenate(e_cat))
    res["mean_within_spearman"] = float(np.mean([v["within_spearman"] for v in res["conditions"].values()]))

    # expert: within-type rho at denoising step 0, by expert layer
    de = M / f"expert_custom_{name}"
    if de.exists():
        E0 = np.load(de / "clean.npz")["expert"].astype(np.float32)
        Rr = np.stack([A.ruler(E0[:, s], ep) for s in range(E0.shape[1])])
        rhos = []
        for c in A.CONDS:
            E = np.load(de / f"{c}.npz")["expert"].astype(np.float32)
            dr = np.linalg.norm(E[:, 0] - E0[:, 0], axis=-1) / Rr[0]  # step 0
            err = np.abs(np.load(d / f"{c}.npz")["actions"] - A0).mean((1, 2))
            rhos.append([A.spearman(dr[:, L], err) for L in range(dr.shape[1])])
        res["expert_within_spearman_step0"] = np.mean(rhos, 0).tolist()

    # patching
    dp = M / f"patch_custom_{name}"
    if dp.exists():
        res["patch"] = {}
        for c in STRONG:
            if not (dp / f"{c}.npz").exists():
                continue  # still running
            p = np.load(dp / f"{c}.npz")
            clean = p["all"]
            base = np.abs(p["none"] - clean).mean((1, 2))
            rec_ = {}
            for k in ["tok_img", "tok_text", "tok_up", "tok_side"] + [f"single_{L}" for L in range(16)]:
                per = np.abs(p[k] - clean).mean((1, 2))
                stat = lambda m: 1 - m[:, 0].mean() / m[:, 1].mean()  # noqa: E731
                both = np.stack([per, base], 1)
                rec_[k] = [float(stat(both)), A.boot_ci(both, ep, stat=stat)]
            odd = np.mean([rec_[f"single_{L}"][0] for L in range(1, 16, 2)])
            even = np.mean([rec_[f"single_{L}"][0] for L in range(0, 16, 2)])
            res["patch"][c] = {"recovery": rec_, "mean_single_odd": float(odd), "mean_single_even": float(even)}
    return res


def main():
    fm = np.load(FEAT / "meta.npz")
    H0 = np.load(FEAT / "clean.npz")["image"][:, 0].astype(np.float32)
    R0 = float(A.ruler(np.load(FEAT / "clean.npz")["image"].astype(np.float32), fm["episode"])[0])
    out = {"split": json.loads((HERE / "split.json").read_text())}
    for name in ("heldout", "seen"):
        out[name] = one_split(name, fm, H0, R0)
        r = out[name]
        print(f"== {name}: {r['n_frames']} frames, episodes {r['episodes'][:5]}...")
        print(f"   clean MAE vs recorded {r['clean_mae_vs_recorded']:.2f} {r['clean_mae_vs_recorded_ci']}")
        print("   action change: " + " ".join(f"{c} {v['action_change']:.2f}" for c, v in r["conditions"].items()))
        print(f"   pooled rho {r['pooled_spearman']:.3f} | mean within {r['mean_within_spearman']:.3f}")
        if "expert_within_spearman_step0" in r:
            e = r["expert_within_spearman_step0"]
            print(f"   expert within rho step0: L0 {e[0]:.2f} L8 {e[8]:.2f} L16 {e[16]:.2f}")
        if "patch" in r:
            for c, v in r["patch"].items():
                rr = v["recovery"]
                print(f"   patch {c}: img {rr['tok_img'][0]:.2f} text {rr['tok_text'][0]:.2f} | single odd {v['mean_single_odd']:.2f} even {v['mean_single_even']:.2f}")
    dref = M / "actions_custom_origheldout"
    if dref.exists():
        am = np.load(dref / "meta.npz")
        A0 = np.load(dref / "clean.npz")["actions"]
        rec = recorded_future(am["index"])
        out["original_model_on_heldout_mae"] = float(np.abs(A0 - rec).mean())
        print(f"original public model (trained on these episodes) MAE vs recorded: {out['original_model_on_heldout_mae']:.2f}")
    # same frames, both models: main-study sub300 frames (6 per episode), split by who trained on them
    dnew = M / "actions_sub300_newmodel"
    if dnew.exists():
        held = set(out["split"]["heldout"])
        m = np.load(ROOT / "runs/drift_study/modal/actions_sub300/meta.npz")
        assert (m["index"] == np.load(dnew / "meta.npz")["index"]).all()
        rec = recorded_future(m["index"])
        orig = np.load(ROOT / "runs/drift_study/modal/actions_sub300/clean.npz")["actions"]
        new = np.load(dnew / "clean.npz")["actions"]
        still = np.repeat(m["joints"][:, None, :], CHUNK, axis=1)
        isheld = np.array([e in held for e in m["episode"]])
        out["same_frames"] = {}
        for lab, mask in (("trained_by_both", ~isheld), ("heldout_from_new", isheld)):
            f = lambda a: float(np.abs(a[mask] - rec[mask]).mean())  # noqa: E731
            out["same_frames"][lab] = {"n": int(mask.sum()), "original": f(orig), "new": f(new), "stay_still": f(still)}
        print("same frames:", out["same_frames"])
    (HERE / "results_heldout.json").write_text(json.dumps(out, indent=1))
    print("saved", HERE / "results_heldout.json")


if __name__ == "__main__":
    main()
