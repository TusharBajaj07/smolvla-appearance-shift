"""SmolVLA drift study (Lesson 3). Same code runs on the Mac (MPS) and on Modal (CUDA).

Modes
  features  VLM prefix pass for every frame x condition (clean + 8 perturbations, see perturb.CONDITIONS).
            Saves per condition: mean image tokens of both cameras [N, 17, 1920] and mean text tokens [N, 17, 960]
            (computed in float32, stored as float16: error ~0.2% of the drift ruler), plus 24x32 pixels of the
            perturbed images (for the pixel-probe control).
  actions   Full model (VLM + 10 flow-matching steps) for every frame x condition with the SAME starting noise per
            frame (seeded by dataset index), so clean vs perturbed differ only by the image. Saves [N, 50, 6] chunks
            in joint units.
  expert    Like actions, but also records the action expert's residual stream (input to each of its 16 layers +
            final norm input = 17 points, mean over the 50 action tokens) at denoising steps 0, 3, 6, 9:
            [N, 4, 17, 720] float16.
  patch     Activation patching on the KV cache the expert reads from the VLM. For each frame: build the clean and
            the perturbed prefix KV cache, then denoise (same noise) with mixed caches:
              none / all          perturbed / clean cache everywhere (sanity: equals perturbed / clean actions)
              single_L (L=0..15)  clean cache at VLM layer L only
              upto_L  (L=0..15)   clean cache at VLM layers 0..L
              tok_up, tok_side, tok_img, tok_text   clean cache for those prefix tokens, all layers
            Saves one [N, 50, 6] action array per recipe.
--seed S changes distractor placement and the starting noise (S=0 reproduces the original run).
Frame sets
  demo32    the 32 frames of runs/probe_pilot/drift_demo.py (for the Mac-vs-GPU check)
  all       every 8th frame of all 50 episodes (1,510)
  sub300    6 frames per episode, spread over each episode (300)
Each condition is a separate file, so an interrupted run resumes where it stopped.

  python study.py --mode features --frames demo32 --limit 8 --out /tmp/x      # smoke test
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import perturb

MODEL = "dajiangw/smolvla_svla_so101_pickplace_finetune"
DATASET = "lerobot/svla_so101_pickplace"
STRIDE, SEED, N_IMG, CHUNK = 8, 0, 64, 50


def pick_frames(ds, which, episodes=None, n_frames=None):
    frame_index = np.array(ds.hf_dataset["frame_index"])
    episode = np.array(ds.hf_dataset["episode_index"])
    all_idx = np.where(frame_index % STRIDE == 0)[0]
    if episodes is not None:  # restrict to a list of episodes (held-out check)
        all_idx = all_idx[np.isin(episode[all_idx], episodes)]
        if which == "custom":
            return all_idx if not n_frames else all_idx[np.linspace(0, len(all_idx) - 1, n_frames).astype(int)]
    if which == "all":
        return all_idx
    if which == "demo32":
        return all_idx[np.linspace(0, len(all_idx) - 1, 32).astype(int)]
    if which == "sub300":
        out = []
        for e in np.unique(episode[all_idx]):
            ep_idx = all_idx[episode[all_idx] == e]
            out.extend(ep_idx[np.linspace(0, len(ep_idx) - 1, 6).astype(int)])
        return np.array(out)
    raise ValueError(which)


class Runner:
    def __init__(self, device, seed=0, model=MODEL):
        self.seed = seed
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

        if device == "cuda":  # full float32 matmuls/convs (TF32 would add ~1e-3 relative error)
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        self.device = device
        self.policy = SmolVLAPolicy.from_pretrained(model).to(device).eval()
        self.pre, self.post = make_pre_post_processors(self.policy.config, pretrained_path=model,
                                                       preprocessor_overrides={"device_processor": {"device": device}})
        model = self.policy.model
        text_model = model.vlm_with_expert.get_vlm_model().text_model
        self.n_layers = model.vlm_with_expert.num_vlm_layers
        self.captured = {}

        def grab(i):
            def hook(_m, args):
                self.captured[i] = args[0].detach()
            return hook

        # residual stream entering each VLM layer + leaving the last one (see probe_pilot/extract.py)
        for i in range(self.n_layers):
            text_model.layers[i].input_layernorm.register_forward_pre_hook(grab(i))
        text_model.norm.register_forward_pre_hook(grab(self.n_layers))

    def batch(self, samples, cond, idxs):
        obs_list, pixels = [], []
        for s, di in zip(samples, idxs):
            obs = {"observation.state": s["observation.state"], "task": s["task"]}
            pix = []
            for c in ("up", "side"):
                img = s[f"observation.images.{c}"]
                if cond is not None:
                    img = perturb.apply(img, *cond, seed=2 * int(di) + (c == "side") + 100000 * self.seed)
                obs[f"observation.images.{c}"] = img
                pix.append(F.adaptive_avg_pool2d(img[None], (24, 32))[0])
            obs_list.append(self.pre(obs))
            pixels.append(torch.cat(pix).flatten().numpy().astype(np.float16))
        b = {k: torch.cat([x[k] for x in obs_list]) for k, v in obs_list[0].items() if torch.is_tensor(v)}
        return b, np.stack(pixels)

    @torch.no_grad()
    def features(self, b):
        from lerobot.policies.smolvla.modeling_smolvla import make_att_2d_masks

        p, m = self.policy, self.policy.model
        images, masks = p.prepare_images(b)
        lang, lmask = b["observation.language.tokens"], b["observation.language.attention_mask"]
        embs, pad, att = m.embed_prefix(images, masks, lang, lmask, state=p.prepare_state(b))
        m.vlm_with_expert.forward(attention_mask=make_att_2d_masks(pad, att), position_ids=torch.cumsum(pad, 1) - 1,
                                  past_key_values=None, inputs_embeds=[embs, None], use_cache=False, fill_kv_cache=True)
        n_lang = lang.shape[1]
        assert embs.shape[1] == 2 * N_IMG + n_lang + 1, embs.shape
        st = torch.stack([self.captured[i] for i in range(self.n_layers + 1)], 1).float()  # [B, 17, T, 960]
        img = torch.cat([st[:, :, :N_IMG].mean(2), st[:, :, N_IMG:2 * N_IMG].mean(2)], -1)
        w = lmask[:, None, :, None].float()
        txt = (st[:, :, 2 * N_IMG:2 * N_IMG + n_lang] * w).sum(2) / w.sum(2)
        return img.cpu().numpy().astype(np.float16), txt.cpu().numpy().astype(np.float16)

    def noise(self, idxs):
        """Starting noise for flow matching, fixed per (frame, seed): identical across conditions and recipes."""
        cfg = self.policy.config
        return torch.stack([torch.randn(CHUNK, cfg.max_action_dim,
                                        generator=torch.Generator().manual_seed(int(i) + 100000 * self.seed))
                            for i in idxs]).to(self.device)

    def enable_expert_hooks(self):
        expert = self.policy.model.vlm_with_expert.lm_expert
        self.expert_cap = []

        def hook(_m, args):
            self.expert_cap.append(args[0].detach().float().mean(1))  # mean over the 50 action tokens -> [B, 720]

        for layer in expert.layers:
            layer.input_layernorm.register_forward_pre_hook(hook)
        expert.norm.register_forward_pre_hook(hook)

    @torch.no_grad()
    def expert_states(self, b, idxs, steps=(0, 3, 6, 9)):
        self.expert_cap = []
        acts = self.actions(b, idxs)
        n = len(self.expert_cap)
        assert n == 10 * (self.n_layers + 1), n  # 10 denoising steps x 17 points
        cap = torch.stack(self.expert_cap).view(10, self.n_layers + 1, len(idxs), -1)  # [step, point, B, 720]
        return cap[list(steps)].permute(2, 0, 1, 3).cpu().numpy().astype(np.float16), acts

    @torch.no_grad()
    def prefix_kv(self, b):
        from lerobot.policies.smolvla.modeling_smolvla import make_att_2d_masks

        p, m = self.policy, self.policy.model
        images, masks = p.prepare_images(b)
        lang, lmask = b["observation.language.tokens"], b["observation.language.attention_mask"]
        embs, pad, att = m.embed_prefix(images, masks, lang, lmask, state=p.prepare_state(b))
        _, kv = m.vlm_with_expert.forward(attention_mask=make_att_2d_masks(pad, att),
                                          position_ids=torch.cumsum(pad, 1) - 1, past_key_values=None,
                                          inputs_embeds=[embs, None], use_cache=True, fill_kv_cache=True)
        return pad, kv, lang.shape[1]

    @torch.no_grad()
    def denoise(self, pad, kv, noise):
        """Same loop as SmolVLA's sample_actions (no RTC), on a given prefix KV cache."""
        m = self.policy.model
        x_t, dt = noise, -1.0 / self.policy.config.num_steps
        for step in range(self.policy.config.num_steps):
            t = torch.tensor(1.0 + step * dt, dtype=torch.float32, device=self.device).expand(noise.shape[0])
            x_t = x_t + dt * m.denoise_step(prefix_pad_masks=pad, past_key_values=kv, x_t=x_t, timestep=t)
        return self.post(x_t[:, :, :6]).float().cpu().numpy()

    @staticmethod
    def mix_kv(kv_pert, kv_clean, layers=None, tokens=None):
        """Perturbed cache with clean values at the given layers (all tokens) or token slice (all layers)."""
        out = {}
        for L, d in kv_pert.items():
            if layers is not None:
                out[L] = kv_clean[L] if L in layers else d
            else:
                out[L] = {}
                for k in ("key_states", "value_states"):
                    t = d[k].clone()
                    t[:, tokens] = kv_clean[L][k][:, tokens]
                    out[L][k] = t
        return out

    @torch.no_grad()
    def patch(self, b_clean, b_pert, idxs):
        noise = self.noise(idxs)
        pad, kv_c, n_lang = self.prefix_kv(b_clean)
        pad_p, kv_p, _ = self.prefix_kv(b_pert)
        assert torch.equal(pad, pad_p)
        n = self.n_layers
        recipes = {"none": kv_p, "all": kv_c}
        for L in range(n):
            recipes[f"single_{L}"] = self.mix_kv(kv_p, kv_c, layers={L})
            recipes[f"upto_{L}"] = self.mix_kv(kv_p, kv_c, layers=set(range(L + 1)))
        recipes["tok_up"] = self.mix_kv(kv_p, kv_c, tokens=slice(0, N_IMG))
        recipes["tok_side"] = self.mix_kv(kv_p, kv_c, tokens=slice(N_IMG, 2 * N_IMG))
        recipes["tok_img"] = self.mix_kv(kv_p, kv_c, tokens=slice(0, 2 * N_IMG))
        recipes["tok_text"] = self.mix_kv(kv_p, kv_c, tokens=slice(2 * N_IMG, 2 * N_IMG + n_lang))
        return {name: self.denoise(pad, kv, noise) for name, kv in recipes.items()}

    @torch.no_grad()
    def actions(self, b, idxs):
        cfg = self.policy.config
        noise = self.noise(idxs)
        self.policy.reset()
        chunk = self.policy.predict_action_chunk(b, noise=noise)
        return self.post(chunk).float().cpu().numpy()[..., :6]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["features", "actions", "expert", "patch"], required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--frames", choices=["demo32", "all", "sub300", "custom"], required=True)
    ap.add_argument("--model", default=MODEL, help="hub id or local checkpoint dir (pretrained_model)")
    ap.add_argument("--episodes", default=None, help="comma list: only frames from these episodes")
    ap.add_argument("--n-frames", type=int, default=None, help="with --frames custom: evenly spaced subset")
    ap.add_argument("--tag", default="", help="suffix for the output folder")
    ap.add_argument("--conditions", default="all", help="'all', 'clean', or comma list like clean,dark_2")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default=None)
    args = ap.parse_args(argv)
    device = args.device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    torch.manual_seed(SEED)
    out = args.out / (f"{args.mode}_{args.frames}" + (f"_s{args.seed}" if args.seed else "") + (f"_{args.tag}" if args.tag else ""))
    out.mkdir(parents=True, exist_ok=True)

    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    ds = LeRobotDataset(DATASET)
    eps = [int(e) for e in args.episodes.split(",")] if args.episodes else None
    idx = pick_frames(ds, args.frames, eps, args.n_frames)[:args.limit]
    conds = [None] + list(perturb.CONDITIONS)
    names = ["clean"] + [f"{n}_{l}" for n, l in perturb.CONDITIONS]
    if args.conditions != "all":
        keep = args.conditions.split(",")
        conds, names = zip(*[(c, n) for c, n in zip(conds, names) if n in keep])

    t_load = time.perf_counter()
    runner = Runner(device, seed=args.seed, model=args.model)
    if args.mode == "expert":
        runner.enable_expert_hooks()
    # metadata from the table only (no video decoding); images are decoded batch by batch below to keep RAM small
    ep_len = {int(e): int(n) for e, n in zip(ds.meta.episodes["episode_index"], ds.meta.episodes["length"])}
    cols = ds.hf_dataset.select(idx.tolist())
    meta = {"index": idx, "episode": np.array(cols["episode_index"]).astype(int),
            "frame": np.array(cols["frame_index"]).astype(int),
            "joints": np.array(cols["observation.state"], dtype=np.float32)}
    meta["progress"] = np.array([f / (ep_len[e] - 1) for e, f in zip(meta["episode"], meta["frame"])])
    np.savez(out / "meta.npz", **meta)
    print(f"{device}: {len(idx)} frames, {len(names)} conditions, setup {time.perf_counter() - t_load:.0f}s", flush=True)

    timing = {}
    for cond, name in zip(conds, names):
        f = out / f"{name}.npz"
        if f.exists():
            print(f"skip {name} (exists)", flush=True)
            continue
        t0 = time.perf_counter()
        if args.mode == "patch" and cond is None:
            continue  # patching compares each perturbation with clean; no file for clean itself
        res = {k: [] for k in {"features": ("image", "text", "pixels"), "actions": ("actions",),
                               "expert": ("expert", "actions"), "patch": ()}[args.mode]}
        for s0 in range(0, len(idx), args.batch):
            b, pix = runner.batch([ds[int(i)] for i in idx[s0:s0 + args.batch]], cond, idx[s0:s0 + args.batch])
            if args.mode == "features":
                img, txt = runner.features(b)
                res["image"].append(img)
                res["text"].append(txt)
                res["pixels"].append(pix)
            elif args.mode == "actions":
                res["actions"].append(runner.actions(b, idx[s0:s0 + args.batch]))
            elif args.mode == "expert":
                ex, ac = runner.expert_states(b, idx[s0:s0 + args.batch])
                res["expert"].append(ex)
                res["actions"].append(ac)
            else:
                b_clean, _ = runner.batch([ds[int(i)] for i in idx[s0:s0 + args.batch]], None, idx[s0:s0 + args.batch])
                for k, v in runner.patch(b_clean, b, idx[s0:s0 + args.batch]).items():
                    res.setdefault(k, []).append(v)
            if (s0 // args.batch) % 40 == 39:
                print(f"  {name}: {s0 + args.batch}/{len(idx)} frames, {time.perf_counter() - t0:.0f}s", flush=True)
        np.savez(f, **{k: np.concatenate(v) for k, v in res.items()})
        timing[name] = time.perf_counter() - t0
        print(f"{name}: {timing[name]:.1f}s ({timing[name] / len(idx):.3f} s/frame)", flush=True)
    (out / f"timing_{device}.json").write_text(json.dumps(timing, indent=1))


if __name__ == "__main__":
    main()
