"""Build every figure and data table of the paper directly from the saved results (no hand-copied numbers).

  .venv-lerobot/bin/python paper/make_figures.py
Inputs : runs/drift_study/{results.json, results_more.json, simpson.json}, runs/probe_pilot/preview_*.jpg
Outputs: paper/figures/*.pdf, paper/tables/*.tex
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RUN, PRE = ROOT / "runs/drift_study", ROOT / "runs/probe_pilot"
FIG, TAB = ROOT / "paper/figures", ROOT / "paper/tables"
FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)

R = json.loads((RUN / "results.json").read_text())
M = json.loads((RUN / "results_more.json").read_text())
S = json.loads((RUN / "simpson.json").read_text())
CONDS = ["dark_1", "dark_2", "blur_1", "blur_2", "distractors_1", "distractors_2", "camera_shift_1", "camera_shift_2"]
STRONG = ["dark_2", "blur_2", "distractors_2", "camera_shift_2"]
COL = {"dark": "#1d6a85", "blur": "#7a8792", "distractors": "#c27c0e", "camera_shift": "#b23b2e"}
NAME = {"dark": "dark", "blur": "blur", "distractors": "clutter", "camera_shift": "shift"}
kind = lambda c: c.rsplit("_", 1)[0]  # noqa: E731
lvl = lambda c: c.rsplit("_", 1)[1]  # noqa: E731
lab = lambda c: f"{NAME[kind(c)]}-{lvl(c)}"  # noqa: E731

plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "legend.fontsize": 7,
                     "axes.spines.top": False, "axes.spines.right": False, "font.family": "serif",
                     "pdf.fonttype": 42, "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5})


def style(c):
    return dict(color=COL[kind(c)], ls="--" if lvl(c) == "1" else "-", lw=1.1 if lvl(c) == "1" else 1.6)


def save(fig, name):
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


# ---- Fig 1: perturbations ----
tiles = ["clean", "dark_1", "dark_2", "blur_1", "blur_2", "distractors_1", "distractors_2", "camera_shift_1", "camera_shift_2"]
fig, axs = plt.subplots(1, 9, figsize=(7.0, 1.05))
for ax, t in zip(axs, tiles):
    ax.imshow(Image.open(PRE / f"preview_{t}.jpg"))
    ax.set_title("clean" if t == "clean" else lab(t), fontsize=7, pad=2)
    ax.axis("off")
fig.subplots_adjust(wspace=0.04)
save(fig, "fig1_perturbations")

# ---- Fig 2: drift and CKA per layer ----
L = np.arange(17)
fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.0))
for c in CONDS:
    v = R["conditions"][c]
    axs[0].plot(L, v["total_drift"], **style(c), label=lab(c))
    axs[1].plot(L[1:], v["update_drift"][1:], **style(c))
    axs[2].plot(L, v["cka"], **style(c))
axs[0].set(title="(a) total drift", xlabel="VLM layer", ylabel="drift (rulers)", ylim=(0, 1.7))
axs[1].set(title="(b) update drift (layers' own additions)", xlabel="VLM layer", ylim=(0, 1.8))
axs[2].set(title="(c) linear CKA with clean", xlabel="VLM layer", ylim=(0.6, 1.0))
fig.tight_layout(rect=(0, 0.1, 1, 1))
fig.legend(*axs[0].get_legend_handles_labels(), ncol=8, frameon=False, loc="lower center", handlelength=1.8, columnspacing=1.0)
save(fig, "fig2_drift")

# ---- Fig 3: transfer, recalibrated transfer, retrained, pixels ----
x = np.arange(len(CONDS))
tr = [R["probe"]["layers"]["16"][c]["gripper_r2"] for c in CONDS]
af = [R["probe"]["layers"]["16"][c]["gripper_r2_affine"] for c in CONDS]
rt = [R["probe_retrained"][c]["L16"]["gripper_r2"] for c in CONDS]
px = [R["probe_retrained"][c]["pixels"]["gripper_r2"] for c in CONDS]
fig, ax = plt.subplots(figsize=(7.0, 1.9))
w = 0.2
series = [(tr, "transfer (fit on clean)", "#1d6a85"), (af, "transfer + affine recalibration", "#8fc1d4"),
          (rt, "retrained on perturbed", "#2c7a4b"), (px, "raw pixels, retrained", "#c27c0e")]
for k, (vals, name, col) in enumerate(series):
    ax.bar(x + (k - 1.5) * w, np.maximum(vals, -1.0), w, color=col, label=name)
    for xi, v in zip(x, vals):
        if v < -1:
            ax.text(xi + (k - 1.5) * w, -0.97, f"{v:.1f}", ha="center", va="bottom", fontsize=6, rotation=90, color="white")
ax.axhline(R["probe"]["layers"]["16"]["clean"]["gripper_r2"], color="k", lw=0.8, ls=":", label="clean")
ax.axhline(0, color="k", lw=0.5)
ax.set(xticks=x, xticklabels=[lab(c) for c in CONDS], ylabel="gripper-opening $R^2$ (layer 16)", ylim=(-1.0, 1.0))
ax.legend(ncol=5, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), columnspacing=1.0, handlelength=1.4)
fig.tight_layout()
save(fig, "fig3_probes")

# ---- Fig 4: Simpson ----
fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.3), gridspec_kw={"width_ratios": [1.5, 1]})
for c in CONDS:
    p = np.array(S["points"][c])
    axs[0].scatter(p[:, 0], p[:, 1], s=2.5, color=COL[kind(c)], alpha=0.30, lw=0)
    axs[0].scatter(p[:, 0].mean(), p[:, 1].mean(), s=30, color=COL[kind(c)], edgecolor="k", lw=0.8, zorder=3,
                   marker="o" if lvl(c) == "2" else "s", label=lab(c))
axs[0].set(title=f"(a) per-frame drift vs action change; pooled $\\rho$ = {S['pooled']:.2f}",
           xlabel="total drift at layer 0 (rulers)", ylabel="action change (joint units)", xlim=(0, 1.9), ylim=(0, 34))
axs[0].legend(ncol=2, frameon=False, fontsize=6, loc="upper left", title="mean per perturbation (square = level 1)",
              title_fontsize=6, handletextpad=0.2, columnspacing=0.6)
ys = np.arange(len(CONDS))[::-1]
for y, c in zip(ys, CONDS):
    rho, (lo, hi) = S["within"][c]
    axs[1].errorbar(rho, y, xerr=[[rho - lo], [hi - rho]], fmt="o", color=COL[kind(c)], ms=3.5, capsize=2, lw=1)
axs[1].errorbar(S["pooled"], -1.2, xerr=[[S["pooled"] - S["pooled_ci"][0]], [S["pooled_ci"][1] - S["pooled"]]],
                fmt="D", color="k", ms=4, capsize=2)
axs[1].axvline(0, color="k", lw=0.6)
axs[1].set(yticks=list(ys) + [-1.2], yticklabels=[lab(c) for c in CONDS] + ["pooled"], xlim=(-0.25, 0.45),
           title="(b) Spearman $\\rho$ within each perturbation", xlabel="$\\rho$ (95% episode-bootstrap CI)")
fig.tight_layout()
save(fig, "fig4_simpson")

# ---- Fig 5: patching ----
P = M["patch"]
fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.2), gridspec_kw={"width_ratios": [1.45, 1, 1]})
groups = [("tok_img", "both cameras"), ("tok_up", "top camera"), ("tok_side", "side camera"), ("tok_text", "instruction")]
gcol = ["#1d6a85", "#7a8792", "#c27c0e", "#b23b2e"]
xs = np.arange(len(STRONG))
for k, (key, name) in enumerate(groups):
    v = np.array([P[c]["recovery"][key] for c in STRONG])
    ci = np.array([P[c]["recovery_ci"][key] for c in STRONG])
    axs[0].bar(xs + (k - 1.5) * 0.2, v, 0.2, color=gcol[k], label=name, yerr=[v - ci[:, 0], ci[:, 1] - v],
               error_kw=dict(lw=0.6, capsize=1))
axs[0].set(xticks=xs, xticklabels=[lab(c) for c in STRONG], ylabel="recovery", ylim=(-0.05, 1.05),
           title="(a) restore KV of token group (all layers)")
axs[0].legend(frameon=False, ncol=4, fontsize=6, loc="lower left", bbox_to_anchor=(0, 1.08), handlelength=1, columnspacing=0.6)
plt.setp(axs[0].get_xticklabels(), rotation=20, ha="right")
for c in STRONG:
    axs[1].plot(range(16), [P[c]["recovery"][f"single_{i}"] for i in range(16)], color=COL[kind(c)], marker="o", ms=2, lw=1.1, label=lab(c))
    axs[2].plot(range(16), [P[c]["recovery"][f"upto_{i}"] for i in range(16)], color=COL[kind(c)], marker="o", ms=2, lw=1.1)
axs[1].set(title="(b) restore a single VLM layer", xlabel="layer (odd = expert cross-attn)", ylim=(-0.05, 1.05), xticks=range(0, 16, 3))
axs[2].set(title="(c) restore layers 0..L", xlabel="L", ylim=(-0.05, 1.05), xticks=range(0, 16, 3))
axs[1].legend(frameon=False, fontsize=6)
fig.tight_layout()
save(fig, "fig5_patching")

# ---- Fig 6: expert ----
E = M["expert"]
fig, axs = plt.subplots(2, 1, figsize=(3.4, 3.6), sharex=True)
for c in STRONG:
    axs[0].plot(L, E["conditions"][c]["drift"][0], color=COL[kind(c)], lw=1.4, label=lab(c))
for s, (step, col) in enumerate(zip(E["steps"], ["#1d6a85", "#7a8792", "#c27c0e", "#b23b2e"])):
    axs[1].plot(L, E["mean_within_spearman"][s], color=col, lw=1.4, label=f"denoising step {step}")
axs[0].set(title="(a) expert drift at denoising step 0", ylabel="drift (rulers)")
axs[1].set(title="(b) within-perturbation $\\rho$(expert drift, action change)", xlabel="action-expert layer", ylabel="Spearman $\\rho$", ylim=(-0.1, 1.0))
axs[0].legend(frameon=False, fontsize=6)
axs[1].legend(frameon=False, fontsize=6)
fig.tight_layout()
save(fig, "fig6_expert")

# ---- Table: per-perturbation summary ----
seeds = M["seeds"]["per_seed"]
rows = []
for c in CONDS:
    v = R["conditions"][c]
    ch = round(100 * (v["update_drift"][16] / v["update_drift"][1] - 1))
    ch_txt = "0" if ch == 0 else f"{ch:+d}"
    acts = [seeds[s]["action_change"][c] for s in ("0", "1", "2")]
    lo, hi = v["total_drift_ci"][0]
    rows.append(f"{lab(c)} & {v['total_drift'][0]:.2f} [{lo:.2f}, {hi:.2f}] & {ch_txt}\\% & {v['cka'][0]:.3f} & "
                f"{R['probe']['layers']['16'][c]['gripper_r2']:.2f} & {R['probe']['layers']['16'][c]['gripper_r2_affine']:.2f} & {R['probe_retrained'][c]['L16']['gripper_r2']:.2f} & "
                f"{np.mean(acts):.2f} $\\pm$ {np.std(acts, ddof=1):.2f} \\\\")
(TAB / "main_table.tex").write_text("\n".join(rows) + "\n\\bottomrule\n")
pr = R["probe"]["layers"]["16"]
macros = {
    "AffBlurTwo": pr["blur_2"]["gripper_r2_affine"], "AffClutterTwo": pr["distractors_2"]["gripper_r2_affine"],
    "RetBlurTwo": R["probe_retrained"]["blur_2"]["L16"]["gripper_r2"], "RetClutterTwo": R["probe_retrained"]["distractors_2"]["L16"]["gripper_r2"],
    "AffDarkOne": pr["dark_1"]["gripper_r2_affine"], "RetDarkOne": R["probe_retrained"]["dark_1"]["L16"]["gripper_r2"],
    "AffShiftOne": pr["camera_shift_1"]["gripper_r2_affine"], "RetShiftOne": R["probe_retrained"]["camera_shift_1"]["L16"]["gripper_r2"],
    "TrBlurTwo": pr["blur_2"]["gripper_r2"], "TrClutterTwo": pr["distractors_2"]["gripper_r2"],
}
mid = ["dark_2", "blur_1", "distractors_1", "camera_shift_2"]
macros["AffMidLo"] = min(pr[c]["gripper_r2_affine"] for c in mid)
macros["AffMidHi"] = max(pr[c]["gripper_r2_affine"] for c in mid)
macros["RetMidLo"] = min(R["probe_retrained"][c]["L16"]["gripper_r2"] for c in mid)
macros["RetMidHi"] = max(R["probe_retrained"][c]["L16"]["gripper_r2"] for c in mid)
(TAB / "numbers.tex").write_text("".join(f"\\newcommand{{\\{k}}}{{{v:.2f}}}\n" for k, v in macros.items()))
print("figures:", sorted(p.name for p in FIG.iterdir()))
print("clean probe", R["probe"]["layers"]["16"]["clean"]["gripper_r2"], "| rulers", [round(R["ruler_total"][i], 1) for i in (0, 16)],
      [round(R["ruler_update"][i], 1) for i in (1, 8, 16)])

# ---- Held-out check (runs/heldout/results_heldout.json) ----
HO_PATH = ROOT / "runs/heldout/results_heldout.json"
if HO_PATH.exists():
    HO = json.loads(HO_PATH.read_text())
    sf = HO["same_frames"]
    def rng(vals):
        return f"{min(vals):.2f}--{max(vals):.2f}"
    rows = []
    def row(label, orig, seen, unseen):
        rows.append(f"{label} & {orig} & {seen} & {unseen} \\\\")
    row("Action MAE vs.\\ recorded (same frames)", f"{sf['trained_by_both']['original']:.2f} / {sf['heldout_from_new']['original']:.2f}$^\\dagger$",
        f"{sf['trained_by_both']['new']:.2f}", f"{sf['heldout_from_new']['new']:.2f}")
    row("``Stay still'' baseline MAE", "--", f"{sf['trained_by_both']['stay_still']:.2f}", f"{sf['heldout_from_new']['stay_still']:.2f}")
    ac = lambda split: rng([v["action_change"] for v in HO[split]["conditions"].values()])  # noqa: E731
    orig_ac = rng([np.mean([seeds[s_]["action_change"][c] for s_ in ("0", "1", "2")]) for c in CONDS])
    row("Action change, range over 8 perturbations", orig_ac, ac("seen"), ac("heldout"))
    row("Pooled / mean within-type $\\rho$", f"{S['pooled']:.2f} / {np.mean([v[0] for v in S['within'].values()]):.2f}",
        f"{HO['seen']['pooled_spearman']:.2f} / {HO['seen']['mean_within_spearman']:.2f}",
        f"{HO['heldout']['pooled_spearman']:.2f} / {HO['heldout']['mean_within_spearman']:.2f}")
    row("Expert-drift $\\rho$ at expert layer 16 (step 0)", f"{M['expert']['mean_within_spearman'][0][16]:.2f}",
        f"{HO['seen']['expert_within_spearman_step0'][16]:.2f}", f"{HO['heldout']['expert_within_spearman_step0'][16]:.2f}")
    img = lambda split: rng([HO[split]["patch"][c]["recovery"]["tok_img"][0] for c in STRONG])  # noqa: E731
    txt = lambda split: rng([HO[split]["patch"][c]["recovery"]["tok_text"][0] for c in STRONG])  # noqa: E731
    odd = lambda split: rng([HO[split]["patch"][c]["mean_single_odd"] for c in STRONG])  # noqa: E731
    evn = lambda split: rng([HO[split]["patch"][c]["mean_single_even"] for c in STRONG])  # noqa: E731
    P0 = M["patch"]
    row("Patching recovery: image tokens", rng([P0[c]["recovery"]["tok_img"] for c in STRONG]), img("seen"), img("heldout"))
    row("Patching recovery: instruction tokens", rng([P0[c]["recovery"]["tok_text"] for c in STRONG]), txt("seen"), txt("heldout"))
    row("Single-layer recovery: odd / even (mean)",
        rng([np.mean([P0[c]["recovery"][f"single_{i}"] for i in range(1, 16, 2)]) for c in STRONG]) + " / " +
        rng([np.mean([P0[c]["recovery"][f"single_{i}"] for i in range(0, 16, 2)]) for c in STRONG]),
        odd("seen") + " / " + evn("seen"), odd("heldout") + " / " + evn("heldout"))
    (TAB / "heldout_table.tex").write_text("\n".join(rows) + "\n\\bottomrule\n")
    ho_macros = {"HoNewSeen": sf["trained_by_both"]["new"], "HoNewHeld": sf["heldout_from_new"]["new"],
                 "HoOrigBoth": sf["trained_by_both"]["original"], "HoOrigHeld": sf["heldout_from_new"]["original"],
                 "HoStillHeld": sf["heldout_from_new"]["stay_still"]}
    with open(TAB / "numbers.tex", "a") as f:
        f.write("".join(f"\\newcommand{{\\{k}}}{{{v:.2f}}}\n" for k, v in ho_macros.items()))
    print("held-out table + macros written")

# ---- Closed loop on LIBERO-Object (runs/libero/results_libero.json, results_patch.json) ----
LB_PATH, LP_PATH = ROOT / "runs/libero/results_libero.json", ROOT / "runs/libero/results_patch.json"
if LB_PATH.exists() and LP_PATH.exists():
    LB, LP = json.loads(LB_PATH.read_text()), json.loads(LP_PATH.read_text())
    pct = lambda v: f"{100 * v:.0f}"  # noqa: E731
    ci = lambda c: f"[{pct(c[0])}, {pct(c[1])}]"  # noqa: E731
    # offline action change of the SO-101 model: mean over the three seeds, the same value as Table I
    seed_mean = lambda c: np.mean([seeds[s_]["action_change"][c] for s_ in ("0", "1", "2")])  # noqa: E731
    rows = []
    for c in ["clean"] + STRONG:
        v = LB["conditions"][c]
        drop = "--" if c == "clean" else f"{pct(v['drop_vs_clean'])} {ci(v['drop_ci'])}"
        off = "--" if c == "clean" else f"{seed_mean(c):.2f}"
        rows.append(f"{'clean' if c == 'clean' else lab(c)} & {pct(v['success'])} {ci(v['ci'])} & {drop} & {off} \\\\")
    PNAME = {"img_all": "image, all layers", "img_odd": "image, odd (cross-attn.)",
             "img_even": "image, even (self-attn.)", "text": "instruction, all layers"}
    A = LP.get("A") or {}
    if A:
        rows.append("\\midrule")
        rows.append("\\multicolumn{4}{l}{\\emph{clutter-2 with clean KV entries for:}} \\\\")
        for r_, v in A.items():
            rows.append(f"\\quad {PNAME[r_]} & {pct(v['success'])} {ci(v['success_ci'])} & "
                        f"{v['success_recovery']:.2f} [{v['success_recovery_ci'][0]:.2f}, {v['success_recovery_ci'][1]:.2f}] & -- \\\\")
    (TAB / "libero_table.tex").write_text("\n".join(rows) + "\n\\bottomrule\n")
    lm = {"LbTasks": str(len(LB["task_ids"])), "LbEps": str(LB["episodes_per_task"]),
          "LbClean": pct(LB["conditions"]["clean"]["success"])}
    for c in STRONG:
        lm["Lb" + NAME[kind(c)].capitalize()] = pct(LB["conditions"][c]["success"])
    if A:
        a0 = next(iter(A.values()))
        lm.update({"PTasks": str(a0["n_tasks"]), "PClean": pct(a0["clean"]), "PClutter": pct(a0["perturbed"])})
    for r_, v in A.items():
        key = {"img_all": "PImg", "img_odd": "POdd", "img_even": "PEven", "text": "PText"}[r_]
        lm[key] = pct(v["success"])
        lm[key + "Rec"] = f"{v['success_recovery']:.2f}"
    OE = LP.get("A_odd_vs_even")
    if OE:  # paired odd- vs even-layer restoration (exact McNemar over episodes, bootstrap over tasks)
        lm.update({"OePairs": str(OE["n_episodes"]), "OeOddOnly": str(OE["odd_only"]), "OeEvenOnly": str(OE["even_only"]),
                   "OeP": f"{OE['mcnemar_p']:.2f}", "OeDiff": pct(OE["diff"]),
                   "OeDiffLo": pct(OE["diff_ci"][0]), "OeDiffHi": pct(OE["diff_ci"][1])})
    B = LP.get("B")
    if B:
        brow = []
        for c in STRONG:
            v = B["conditions"][c]
            rr = v["recovery"]
            brow.append(f"{lab(c)} & {v['drift_by_layer'][0]:.2f} / {v['drift_by_layer'][-1]:.2f} & {v['action_change']:.3f} & "
                        f"{rr['img_all'][0]:.2f} & {rr['img_odd'][0]:.2f} & {rr['img_even'][0]:.2f} & {rr['text'][0]:.2f} & "
                        f"{v['mean_single_odd']:.2f} / {v['mean_single_even']:.2f} & {v['within_spearman']:.2f} \\\\")
        (TAB / "libero_mech_table.tex").write_text("\n".join(brow) + "\n\\bottomrule\n")
        lm.update({"LmFrames": f"{B['n_frames']:,}", "LmEpisodes": str(B["n_episodes"]), "LmLayers": str(B["n_layers"]),
                   "LmPooled": f"{B['pooled_spearman']:.2f}", "LmWithin": f"{B['mean_within_spearman']:.2f}"})
    with open(TAB / "numbers.tex", "a") as f:
        f.write("".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in lm.items()))
    print("LIBERO tables + macros written:", lm)
