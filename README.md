# Moved, Not Lost: locating appearance-shift failures inside SmolVLA

Code, result files and paper source for *"Moved, Not Lost: Locating Appearance-Shift Failures Inside a Small
Vision-Language-Action Model"* (Tushar Bajaj, IIT Bombay).

Vision-language-action (VLA) robot policies fail when lighting, focus, clutter or camera placement change.
Benchmarks report how often. This project traces *where inside the model* the failure arises, using
[SmolVLA](https://arxiv.org/abs/2506.01844) (450M parameters) and eight controlled image perturbations
(dark, blur, clutter, camera shift; two strengths each).

Main findings (details and confidence intervals in the paper, `paper/main.pdf`):

- Drift is set at the vision encoder and passes through the frozen language backbone essentially unchanged.
- Task information (gripper opening) survives but is re-encoded: probes retrained on perturbed features decode it
  (R² 0.71–0.91) while clean-fitted probes fail.
- Per-frame drift predicts which *kind* of perturbation hurts, not which *frame* fails (a Simpson's paradox).
- Activation patching on the KV cache: the action change travels through image-token keys and values, read by the
  action expert's cross-attention layers; instruction tokens carry none of it.
- In closed loop (LIBERO-Object), the perturbations cut success from 88% to 6-74%, in the same order as the offline
  action changes; restoring only the image tokens' cache entries at every control step under clutter raises success
  from 7% to 80% (6 tasks x 5 episodes, one seed).

## Layout

| Path | What it is |
|---|---|
| `runs/drift_study/` | Offline study on SO-101 pick-and-place: features, actions, action-expert states, KV patching (`study.py`), Modal launcher, analyses, results JSON |
| `runs/heldout/` | Held-out replication: fine-tune on 40 of 50 episodes (`train_app.py`), analysis, results |
| `runs/libero/` | Closed-loop LIBERO evaluation with perturbations, closed-loop KV patching and in-rollout measurements (`eval_perturbed.py`), Modal launcher, analyses, results JSON, and `episodes.json` (success of every episode in every run) |
| `runs/probe_pilot/` | Example perturbed frames used in Fig. 1 |
| `paper/` | LaTeX source; `make_figures.py` regenerates every figure, table and number from the result files |

## Reproduce

Environment (pinned versions used for every result):

```sh
python3.10 -m venv .venv && . .venv/bin/activate
pip install torch==2.10.0 torchvision==0.25.0 "lerobot[smolvla]==0.4.4" modal matplotlib
modal setup   # GPU runs use Modal (NVIDIA L4 / A10G for analysis and evaluation, H100 for the held-out fine-tune)
```

Offline study (outputs go to the Modal volume `smolvla-drift`; download them to `runs/drift_study/modal/`):

```sh
modal run runs/drift_study/modal_app.py --mode features --frames all
modal run runs/drift_study/modal_app.py --mode actions  --frames all          # add --seed 1 / --seed 2
modal run runs/drift_study/modal_app.py --mode expert   --frames sub300
modal run runs/drift_study/modal_app.py --mode patch    --frames sub300
python runs/drift_study/analyze.py && python runs/drift_study/analyze_more.py
```

Held-out check: `modal run runs/heldout/train_app.py --steps 15000 --gpu H100 --amp`, then the `custom` frame sets
described in `runs/heldout/analyze_heldout.py`.

Closed loop (official `HuggingFaceVLA/smolvla_libero` checkpoint, LIBERO-Object, 10 tasks × 5 episodes, seed 0):

```sh
modal run --detach runs/libero/libero_app.py --cond distractors_2 --n-episodes 5 --batch 5           # one condition
modal run --detach runs/libero/libero_app.py --cond distractors_2 --patch img_all --n-episodes 5 --batch 5
modal run --detach runs/libero/libero_app.py --cond clean --measure --n-episodes 5 --batch 5
# download the run outputs from the Modal volume to runs/libero/modal/, then:
python runs/libero/analyze_libero.py && python runs/libero/analyze_patch.py
```

Paper: `python paper/make_figures.py && cd paper && latexmk -pdf main.tex`.

## License

Code: MIT (see `LICENSE`). SmolVLA, LeRobot, LIBERO and the SO-101 dataset are third-party and keep their own
licenses.
