"""Analyse the Modal drift-study outputs (CPU only). Writes runs/drift_study/results.json.

A. drift per layer and condition: total (h_L) and update (u_L = h_L - h_0), in "rulers" = median distance between
   two clean frames from DIFFERENT episodes; mean with 95% CI from an episode bootstrap.
B. linear CKA per layer between clean and perturbed features (does the arrangement of frames survive?).
C. probes under perturbation: ridge probes trained on CLEAN frames, tested on clean and perturbed frames of held-out
   episodes (5-fold by episode), image tokens per layer and the raw-pixel control. Also reports the transfer probe's R^2
   after the best affine recalibration of its outputs (= squared correlation) and after a shift only, to separate a
   simple offset of the representation from a genuine change of the readout direction.
D. action error: mean |actions(perturbed) - actions(clean)| over the 50-step chunk and 6 joints (same noise).
E. per layer, Spearman correlation across perturbed frames between drift and action error, pooled over the
   8 conditions, with episode-bootstrap CI; control = pixel-space drift.

  .venv-lerobot/bin/python runs/drift_study/analyze.py
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "probe_pilot"))
import probe as P  # noqa: E402  (ridge_predict_all, standardize, r2, balanced_acc, LAMBDAS)

FEAT = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "modal/features_all"
ACT = Path(sys.argv[2]) if len(sys.argv) > 2 else HERE / "modal/actions_sub300"
OUT = Path(sys.argv[3]) if len(sys.argv) > 3 else HERE / "results.json"
CONDS = ["dark_1", "dark_2", "blur_1", "blur_2", "distractors_1", "distractors_2", "camera_shift_1", "camera_shift_2"]
N_BOOT, SEED, K = 1000, 0, 5
OPEN_THRESHOLD = 10.0
rng = np.random.default_rng(SEED)


def boot_ci(per_frame, episodes, stat=np.mean):
    """95% CI of a statistic by resampling whole episodes (frames within an episode are not independent)."""
    eps = np.unique(episodes)
    groups = [np.where(episodes == e)[0] for e in eps]
    vals = []
    for _ in range(N_BOOT):
        pick = np.concatenate([groups[i] for i in rng.integers(0, len(eps), len(eps))])
        vals.append(stat(per_frame[pick]))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def ruler(X, episodes, n_pairs=20000):
    """Median distance between two clean frames from different episodes, per layer. X: [N, 17, D]."""
    i, j = rng.integers(0, len(X), (2, n_pairs))
    keep = episodes[i] != episodes[j]
    i, j = i[keep], j[keep]
    return np.median(np.linalg.norm(X[i] - X[j], axis=-1), axis=0)  # [17]


def linear_cka(X, Y):
    X, Y = X - X.mean(0), Y - Y.mean(0)
    return float(np.linalg.norm(Y.T @ X) ** 2 / (np.linalg.norm(X.T @ X) * np.linalg.norm(Y.T @ Y)))


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    meta = np.load(FEAT / "meta.npz")
    ep, idx = meta["episode"], meta["index"]
    gripper = meta["joints"][:, 5].astype(np.float64)
    clean = np.load(FEAT / "clean.npz")
    H0 = clean["image"].astype(np.float32)                  # [N, 17, 1920]
    U0 = H0 - H0[:, :1]
    pix0 = clean["pixels"].astype(np.float32)
    R_tot, R_upd = ruler(H0, ep), ruler(U0, ep)
    R_upd[0] = 1.0
    R_pix = float(ruler(pix0[:, None], ep)[0])
    res = {"n_frames": int(len(ep)), "n_episodes": int(len(np.unique(ep))), "conditions": {},
           "ruler_total": R_tot.tolist(), "ruler_update": R_upd.tolist()}
    per_frame_drift = {}

    # ---- A, B: drift and CKA ----
    for c in CONDS:
        d = np.load(FEAT / f"{c}.npz")
        H = d["image"].astype(np.float32)
        tot = np.linalg.norm(H - H0, axis=-1) / R_tot      # [N, 17]
        upd = np.linalg.norm((H - H[:, :1]) - U0, axis=-1) / R_upd
        pixd = np.linalg.norm(d["pixels"].astype(np.float32) - pix0, axis=-1) / R_pix
        per_frame_drift[c] = (tot, upd, pixd)
        res["conditions"][c] = {
            "total_drift": tot.mean(0).tolist(),
            "total_drift_ci": [boot_ci(tot[:, L], ep) for L in (0, 8, 16)],
            "update_drift": upd.mean(0).tolist(),
            "update_drift_ci": [boot_ci(upd[:, L], ep) for L in (1, 8, 16)],
            "pixel_drift": float(pixd.mean()),
            "cka": [linear_cka(H0[:, L].astype(np.float64), H[:, L].astype(np.float64)) for L in range(17)],
        }
        print(f"{c:15s} total L0 {tot.mean(0)[0]:.2f} L16 {tot.mean(0)[16]:.2f} | update L1 {upd.mean(0)[1]:.2f} "
              f"L8 {upd.mean(0)[8]:.2f} L16 {upd.mean(0)[16]:.2f} | CKA L0 {res['conditions'][c]['cka'][0]:.3f} "
              f"L16 {res['conditions'][c]['cka'][16]:.3f}", flush=True)

    # ---- C: probes trained on clean, tested on perturbed held-out episodes ----
    eps = rng.permutation(np.unique(ep))
    fold = np.array([{e: i % K for i, e in enumerate(eps)}[e] for e in ep])
    Y = np.stack([gripper, (gripper > OPEN_THRESHOLD).astype(np.float64)], 1)
    feats_all = {c: np.load(FEAT / f"{c}.npz") for c in CONDS}
    sets = ["clean"] + CONDS

    def probe_scores(get):  # get(cond) -> [N, D] features
        preds = {s: np.zeros_like(Y) for s in sets}
        Xs = {s: get(s) for s in sets}
        for k in range(K):
            tr, te = fold != k, fold == k
            g_tr = ep[tr]
            val = np.isin(g_tr, rng.choice(np.unique(g_tr), len(np.unique(g_tr)) // 5, replace=False))
            Xtr = Xs["clean"][tr].astype(np.float64)
            err = ((P.ridge_predict_all(Xtr[~val], Y[tr][~val], Xtr[val]) - Y[tr][val]) ** 2).mean(1)
            best = err.argmin(0)
            # one fit on clean training frames, applied to every condition's held-out frames
            Xte = np.concatenate([Xs[s][te] for s in sets]).astype(np.float64)
            Pall = P.ridge_predict_all(Xtr, Y[tr], Xte)[best, :, np.arange(2)].T
            n = te.sum()
            for si, s in enumerate(sets):
                preds[s][te] = Pall[si * n:(si + 1) * n]
        def recal(y, p):  # best affine recalibration of the clean readout = squared Pearson correlation
            return float(np.corrcoef(p, y)[0, 1] ** 2)

        return {s: {"gripper_r2": P.r2(Y[:, 0], preds[s][:, 0]),
                    "gripper_r2_affine": recal(Y[:, 0], preds[s][:, 0]),
                    "gripper_r2_shift": P.r2(Y[:, 0], preds[s][:, 0] - preds[s][:, 0].mean() + Y[:, 0].mean()),
                    "open_bacc": P.balanced_acc(Y[:, 1], preds[s][:, 1])} for s in sets}

    res["probe"] = {"layers": {}}
    for L in range(17):
        res["probe"]["layers"][L] = probe_scores(
            lambda s, L=L: (H0 if s == "clean" else feats_all[s]["image"])[:, L].astype(np.float32))
        sc = res["probe"]["layers"][L]
        print(f"probe L{L:2d}: clean R2 {sc['clean']['gripper_r2']:.3f} | " +
              " ".join(f"{c} {sc[c]['gripper_r2']:.2f}" for c in CONDS), flush=True)
    res["probe"]["pixels"] = probe_scores(lambda s: (pix0 if s == "clean" else feats_all[s]["pixels"]).astype(np.float32))
    print("probe pixels: " + " ".join(f"{s} {v['gripper_r2']:.2f}" for s, v in res["probe"]["pixels"].items()), flush=True)

    # Retrained probes: trained AND tested on perturbed frames (episode CV). Transfer score low but retrained score
    # high = information still present, just encoded differently; both low = information lost.
    def within(X):
        Pr = P.cross_validate(X.astype(np.float64), Y, ep, "episode", SEED)
        return {"gripper_r2": P.r2(Y[:, 0], Pr[:, 0]), "open_bacc": P.balanced_acc(Y[:, 1], Pr[:, 1])}

    res["probe_retrained"] = {}
    for c in CONDS:
        r = {f"L{L}": within(feats_all[c]["image"][:, L]) for L in (0, 8, 16)}
        r["pixels"] = within(feats_all[c]["pixels"])
        res["probe_retrained"][c] = r
        print(f"retrained {c:15s} " + " ".join(f"{k} {v['gripper_r2']:.2f}" for k, v in r.items()), flush=True)

    # ---- D, E: action error and its link to drift ----
    am = np.load(ACT / "meta.npz")
    pos = {int(i): k for k, i in enumerate(idx)}
    rows = np.array([pos[int(i)] for i in am["index"]])  # sub300 frames inside the 1,510
    A0 = np.load(ACT / "clean.npz")["actions"]
    act_err, d_tot, d_upd, d_pix, ep_rows = [], [], [], [], []
    res["actions"] = {}
    for c in CONDS:
        A = np.load(ACT / f"{c}.npz")["actions"]
        e = np.abs(A - A0).mean((1, 2))                     # [300]
        res["actions"][c] = {"mean_abs_change": float(e.mean()), "ci": boot_ci(e, am["episode"]),
                             "per_joint": np.abs(A - A0).mean((0, 1)).tolist()}
        tot, upd, pixd = per_frame_drift[c]
        act_err.append(e)
        d_tot.append(tot[rows])
        d_upd.append(upd[rows])
        d_pix.append(pixd[rows])
        ep_rows.append(am["episode"])
        print(f"actions {c:15s} mean |change| {e.mean():.2f} joint units", flush=True)
    act_err, d_tot, d_upd = np.concatenate(act_err), np.concatenate(d_tot), np.concatenate(d_upd)
    d_pix, ep_rows = np.concatenate(d_pix), np.concatenate(ep_rows)

    def corr_ci(x):
        both = np.stack([x, act_err], 1)
        return boot_ci(both, ep_rows, stat=lambda m: spearman(m[:, 0], m[:, 1]))

    res["correlation"] = {
        "total_drift": [spearman(d_tot[:, L], act_err) for L in range(17)],
        "update_drift": [spearman(d_upd[:, L], act_err) if L else None for L in range(17)],
        "total_drift_ci": {L: corr_ci(d_tot[:, L]) for L in (0, 8, 16)},
        "update_drift_ci": {L: corr_ci(d_upd[:, L]) for L in (1, 8, 16)},
        "pixel_drift": spearman(d_pix, act_err), "pixel_drift_ci": corr_ci(d_pix),
        "n_pairs": int(len(act_err)),
    }
    c = res["correlation"]
    print(f"Spearman(drift, action change) over {c['n_pairs']} perturbed frames: total L0 {c['total_drift'][0]:.3f} "
          f"L16 {c['total_drift'][16]:.3f} | update L1 {c['update_drift'][1]:.3f} L8 {c['update_drift'][8]:.3f} "
          f"L16 {c['update_drift'][16]:.3f} | pixels {c['pixel_drift']:.3f}", flush=True)
    OUT.write_text(json.dumps(res, indent=1))
    print("saved", OUT)


if __name__ == "__main__":
    main()
