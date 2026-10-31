"""LeRobot's official LIBERO evaluation (lerobot-eval), with one image perturbation applied to both cameras.

Only three things of lerobot.scripts.lerobot_eval / SmolVLA are wrapped; everything else (environment, policy,
processors, success criterion, episode seeds, videos) is the official code path:
  preprocess_observation  after the official conversion to float [B,3,H,W] in [0,1], apply perturb.apply to every
                          camera image of every parallel environment (and keep the clean images for PATCH/MEASURE)
  rollout                 count episodes so each episode gets its own, but within-episode constant, clutter layout
  SmolVLAPolicy._get_action_chunk   only when PATCH or MEASURE is set (see below)

  PERTURB=blur_2 PERTURB_SEED=0 python eval_perturbed.py --policy.path=HuggingFaceVLA/smolvla_libero \
      --env.type=libero --env.task=libero_object --eval.n_episodes=10 --eval.batch_size=10 --output_dir=out
PERTURB=clean runs the unmodified evaluation.

Experiment A, closed-loop activation patching (PATCH=img_all|img_odd|img_even|text): the policy sees the perturbed
images, but before the action expert reads the VLM key-value cache, the cache entries of the chosen prefix tokens are
replaced by the ones computed from the clean images of the same simulator state (all 16 layers, or only the odd
layers the expert reads through cross-attention, or only the even self-attention layers). Same noise for both.

Experiment B, mechanism on this checkpoint (MEASURE=<dir>, PERTURB=clean): the policy acts on clean images; every
MEASURE_EVERY control steps, each strong perturbation is applied to the current frame and we save the per-layer mean
image/text residual stream (clean and perturbed) and the action chunks under several KV-patching recipes, all with
one shared noise sample. Frames of environments whose episode already ended are flagged (valid=False).
"""
import copy
import os
from pathlib import Path

import numpy as np
import torch

import perturb
from lerobot.policies.smolvla import modeling_smolvla as MS
from lerobot.scripts import lerobot_eval as E

COND = os.environ.get("PERTURB", "clean")
SEED = int(os.environ.get("PERTURB_SEED", "0"))
PATCH = os.environ.get("PATCH", "")
MEASURE = os.environ.get("MEASURE", "")
MEASURE_EVERY = int(os.environ.get("MEASURE_EVERY", "20"))
STRONG = ["dark_2", "blur_2", "distractors_2", "camera_shift_2"]
_episode_batch = [0]
_ctx = {}  # env, processors, clean observation, step counter, done flags of the current rollout
_orig_pre, _orig_rollout, _orig_chunk = E.preprocess_observation, E.rollout, MS.SmolVLAPolicy._get_action_chunk


def _perturb(obs, cond):
    name, level = cond.rsplit("_", 1)
    keys = sorted(k for k in obs if k.startswith("observation.images."))
    for ci, key in enumerate(keys):
        imgs = obs[key]  # [B, 3, H, W], float in [0, 1]
        obs[key] = torch.stack([
            perturb.apply(imgs[b], name, int(level), seed=100000 * SEED + 1000 * _episode_batch[0] + 10 * b + ci)
            for b in range(imgs.shape[0])])
    return obs


def _rollout(**kwargs):
    _episode_batch[0] += 1
    env = kwargs["env"]
    _ctx.update(env=env, env_pre=kwargs["env_preprocessor"], pre=kwargs["preprocessor"], step=0,
                done=np.zeros(env.num_envs, bool))
    step = env.step

    def tracked_step(action):
        out = step(action)
        _ctx["done"] |= np.asarray(out[2]) | np.asarray(out[3])  # terminated | truncated
        return out

    env.step = tracked_step
    try:
        return _orig_rollout(**kwargs)
    finally:
        env.step = step


def _pre(observation):
    out = _orig_pre(observation)
    if PATCH or MEASURE:
        _ctx["clean"] = copy.deepcopy(out)
    if COND == "clean":
        return out
    return _perturb(out, COND)


def _prep(policy, obs):
    """Same steps as rollout(): task string, LIBERO env processor, policy preprocessor, observation queue shape."""
    b = _ctx["pre"](_ctx["env_pre"](E.add_envs_task(_ctx["env"], copy.deepcopy(obs))))
    for k in b:
        if k in policy._queues and k != MS.ACTION:
            b[k] = b[k][:, None]  # n_obs_steps = 1: the queue holds exactly the current observation
    return b


def _prefix(policy, batch):
    m = policy.model
    images, masks = policy.prepare_images(batch)
    lang, lmask = batch[MS.OBS_LANGUAGE_TOKENS], batch[MS.OBS_LANGUAGE_ATTENTION_MASK]
    embs, pad, att = m.embed_prefix(images, masks, lang, lmask, state=policy.prepare_state(batch))
    _, kv = m.vlm_with_expert.forward(attention_mask=MS.make_att_2d_masks(pad, att),
                                      position_ids=torch.cumsum(pad, 1) - 1, past_key_values=None,
                                      inputs_embeds=[embs, None], use_cache=True, fill_kv_cache=True)
    n_lang = lang.shape[1]
    n_img = embs.shape[1] - n_lang - 1  # prefix = image tokens of every camera, instruction, 1 state token
    return pad, kv, n_img, n_lang, lmask


def _denoise(policy, pad, kv, noise):
    """SmolVLA's sample_actions loop (no RTC) on a given prefix KV cache."""
    m, n = policy.model, policy.config.num_steps
    x, dt = noise, -1.0 / n
    for s in range(n):
        t = torch.tensor(1.0 + s * dt, dtype=torch.float32, device=noise.device).expand(noise.shape[0])
        x = x + dt * m.denoise_step(prefix_pad_masks=pad, past_key_values=kv, x_t=x, timestep=t)
    return x


def _mix(kv_p, kv_c, tokens, layers):
    out = {}
    for L, d in kv_p.items():
        if L not in layers:
            out[L] = d
            continue
        out[L] = {}
        for k in ("key_states", "value_states"):
            t = d[k].clone()
            t[:, tokens] = kv_c[L][k][:, tokens]
            out[L][k] = t
    return out


def _recipe(name, n_img, n_lang, n_layers):
    """(token slice, layer set) of a patching recipe."""
    img, txt, every = slice(0, n_img), slice(n_img, n_img + n_lang), set(range(n_layers))
    if name == "img_all":
        return img, every
    if name == "img_odd":  # layers the expert reads through cross-attention
        return img, {L for L in every if L % 2 == 1}
    if name == "img_even":  # layers where the expert self-attends over [prefix, actions]
        return img, {L for L in every if L % 2 == 0}
    if name == "text":
        return txt, every
    if name == "all":
        return slice(None), every
    if name.startswith("single_"):
        return slice(None), {int(name.split("_")[1])}
    raise ValueError(name)


_checked = [False]


@torch.no_grad()
def _patched_chunk(self, batch, noise=None, **kwargs):
    for k in batch:  # as in the original _get_action_chunk
        if k in self._queues and k != MS.ACTION:
            batch[k] = torch.stack(list(self._queues[k]), dim=1)
    cfg, dev = self.config, batch[MS.OBS_STATE].device
    B = batch[MS.OBS_STATE].shape[0]
    noise = self.model.sample_noise((B, cfg.chunk_size, cfg.max_action_dim), dev) if noise is None else noise
    pad, kv_p, n_img, n_lang, _ = _prefix(self, batch)
    b_c = _prep(self, _ctx["clean"])
    pad_c, kv_c, *_ = _prefix(self, b_c)
    assert torch.equal(pad, pad_c)
    tokens, layers = _recipe(PATCH, n_img, n_lang, len(kv_p))
    x = _denoise(self, pad, _mix(kv_p, kv_c, tokens, layers), noise)
    if not _checked[0]:  # exactness checks, once per run
        _checked[0] = True
        images, masks = self.prepare_images(batch)
        ref = self.model.sample_actions(images, masks, batch[MS.OBS_LANGUAGE_TOKENS],
                                        batch[MS.OBS_LANGUAGE_ATTENTION_MASK], self.prepare_state(batch), noise=noise)
        own = _denoise(self, pad, kv_p, noise)
        allc = _denoise(self, pad, _mix(kv_p, kv_c, slice(None), set(range(len(kv_p)))), noise)
        print(f"PATCH_CHECK recipe={PATCH} n_img={n_img} n_lang={n_lang} layers={sorted(layers)} "
              f"|own-official|={(own - ref).abs().max():.2e} |all-clean|={(allc - _denoise(self, pad_c, kv_c, noise)).abs().max():.2e} "
              f"|patched-pert|={(x - own).abs().mean():.3e}", flush=True)
    return x[:, :, :cfg.action_feature.shape[0]]


def _hooks(policy):
    if "cap" in _ctx:
        return
    tm = policy.model.vlm_with_expert.get_vlm_model().text_model
    _ctx["cap"] = {}
    n = policy.model.vlm_with_expert.num_vlm_layers

    def grab(i):
        def hook(_m, args):
            _ctx["cap"][i] = args[0].detach()
        return hook

    for i in range(n):
        tm.layers[i].input_layernorm.register_forward_pre_hook(grab(i))
    tm.norm.register_forward_pre_hook(grab(n))
    _ctx["n_points"] = n + 1


def _feats(n_img, n_lang, lmask):
    st = torch.stack([_ctx["cap"][i] for i in range(_ctx["n_points"])], 1).float()  # [B, 17, T, 960]
    img = st[:, :, :n_img].mean(2)
    w = lmask[:, None, :, None].float()
    txt = (st[:, :, n_img:n_img + n_lang] * w).sum(2) / w.sum(2)
    return img.cpu().numpy().astype(np.float16), txt.cpu().numpy().astype(np.float16)


@torch.no_grad()
def _measure(self, batch):
    _hooks(self)
    cfg, B = self.config, batch[MS.OBS_STATE].shape[0]
    dev = batch[MS.OBS_STATE].device
    g = torch.Generator(device="cpu").manual_seed(100000 * SEED + 1000 * _episode_batch[0] + _ctx["step"])
    noise = torch.randn(B, cfg.chunk_size, cfg.max_action_dim, generator=g).to(dev)
    d = cfg.action_feature.shape[0]
    pad, kv_c, n_img, n_lang, lmask = _prefix(self, _prep(self, _ctx["clean"]))
    fi, ft = _feats(n_img, n_lang, lmask)
    rec = {"step": np.full(B, _ctx["step"]), "valid": ~_ctx["done"], "env": np.arange(B),
           "feat_img_clean": fi, "feat_txt_clean": ft,
           "act_clean": _denoise(self, pad, kv_c, noise)[:, :, :d].cpu().numpy()}
    n_layers = len(kv_c)
    recipes = ["img_all", "img_odd", "img_even", "text"] + [f"single_{L}" for L in range(n_layers)]
    for cond in STRONG:
        b_p = _prep(self, _perturb(copy.deepcopy(_ctx["clean"]), cond))
        pad_p, kv_p, *_ = _prefix(self, b_p)
        assert torch.equal(pad, pad_p)
        fi, ft = _feats(n_img, n_lang, lmask)
        rec[f"feat_img_{cond}"], rec[f"feat_txt_{cond}"] = fi, ft
        rec[f"act_{cond}_none"] = _denoise(self, pad, kv_p, noise)[:, :, :d].cpu().numpy()
        for r in recipes:
            tok, lay = _recipe(r, n_img, n_lang, n_layers)
            rec[f"act_{cond}_{r}"] = _denoise(self, pad, _mix(kv_p, kv_c, tok, lay), noise)[:, :, :d].cpu().numpy()
    Path(MEASURE).mkdir(parents=True, exist_ok=True)  # one file per measurement: a killed run keeps what it measured
    np.savez_compressed(f"{MEASURE}/batch{_episode_batch[0]}_step{_ctx['step']:03d}.npz", **rec)


def _chunk(self, batch, noise=None, **kwargs):
    if MEASURE:
        if _ctx["step"] % MEASURE_EVERY == 0:
            _measure(self, batch)
        _ctx["step"] += 1
        return _orig_chunk(self, batch, noise, **kwargs)
    return _patched_chunk(self, batch, noise, **kwargs)


E.preprocess_observation = _pre
E.rollout = _rollout
if PATCH or MEASURE:
    MS.SmolVLAPolicy._get_action_chunk = _chunk

if __name__ == "__main__":
    print(f"PERTURB={COND} PERTURB_SEED={SEED} PATCH={PATCH or '-'} MEASURE={MEASURE or '-'}", flush=True)
    E.eval_main()
