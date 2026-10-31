"""Controlled image perturbations for the drift study (Lesson 3).

Every function takes an image tensor [3, H, W] in [0, 1] (LeRobotDataset format) and returns one of the same shape.
Each perturbation has two severities. Randomness (where distractors go) is seeded per frame, so reruns are identical.
Applied to BOTH cameras: only the appearance changes, never the robot state or the instruction.
Pixel-sized parameters (blur sigma, block sizes) are defined for 480-pixel-high frames and scale with image height
(scale = H / 480), so a 256x256 LIBERO frame gets the same relative severity. At H = 480 nothing changes.
"""
import math

import torch
import torchvision.transforms.functional as TF

CONDITIONS = [  # (name, severity) -> label used in results
    ("dark", 1), ("dark", 2),
    ("blur", 1), ("blur", 2),
    ("distractors", 1), ("distractors", 2),
    ("camera_shift", 1), ("camera_shift", 2),
]


def dark(img, level):
    """Dim the scene: multiply every pixel by 0.6 (level 1) or 0.35 (level 2)."""
    return img * {1: 0.6, 2: 0.35}[level]


def blur(img, level, scale=1.0):
    """Gaussian blur with sigma 2 px (level 1) or 5 px (level 2) on a 480-pixel-high frame (out of focus / motion)."""
    sigma = {1: 2.0, 2: 5.0}[level] * scale
    k = 2 * math.ceil(3 * sigma) + 1
    return TF.gaussian_blur(img, kernel_size=[k, k], sigma=[sigma, sigma])


def distractors(img, level, seed, scale=1.0):
    """Paste 3 (level 1) or 8 (level 2) solid-coloured rectangles, 30-70 px, at random places (clutter).
    Limitation: synthetic blobs, not real objects; they may also cover the gripper or brick (checked via pixel probe)."""
    g = torch.Generator().manual_seed(seed)
    out = img.clone()
    _, H, W = img.shape
    for _ in range({1: 3, 2: 8}[level]):
        lo, hi = max(1, round(30 * scale)), max(2, round(70 * scale))
        h, w = (int(torch.randint(lo, hi + 1, (1,), generator=g)) for _ in range(2))
        y, x = int(torch.randint(0, H - h, (1,), generator=g)), int(torch.randint(0, W - w, (1,), generator=g))
        out[:, y:y + h, x:x + w] = torch.rand(3, 1, 1, generator=g)
    return out


def camera_shift(img, level):
    """Simulated camera bump: shift 5% / 10% of the frame, rotate 3 / 6 degrees, zoom 1.05 / 1.10.
    A 2D approximation: a real camera move also changes perspective (parallax), which this cannot reproduce."""
    _, H, W = img.shape
    f, ang, zoom = {1: (0.05, 3.0, 1.05), 2: (0.10, 6.0, 1.10)}[level]
    return TF.affine(img, angle=ang, translate=[int(f * W), int(f * H)], scale=zoom, shear=[0.0], fill=0.0)


def apply(img, name, level, seed=0):
    scale = img.shape[-2] / 480.0
    if name == "dark":
        return dark(img, level)
    if name == "blur":
        return blur(img, level, scale)
    if name == "distractors":
        return distractors(img, level, seed, scale)
    if name == "camera_shift":
        return camera_shift(img, level)
    raise ValueError(name)
