"""Data-derived posture prior for IK branch selection.

A redundant arm reaches the same end-effector pose with a continuum of joint configurations.
cuRobo ranks its IK seeds by ``pose_error + null_space_error`` with ``null_space_cfg.weight`` at
0.001 against a generic home retract, so with the default ``return_seeds=1`` the branch is picked
by pose error alone -- and the arm's redundancy lands wherever, independently at every endpoint.
The result is a planner that visits joint configurations human teleoperation never does.

This module scores a configuration by how far its joint-PAIR differences fall outside the range
humans use, weighted by how much each pair's direction is actually free to move:

    penalty(q) = sum_{i<j}  w_ij * [ relu(d_ij - hi_ij) + relu(lo_ij - d_ij) ],   d_ij = q_i - q_j

Both halves come from data, nothing is hand-picked:

* ``lo_ij`` / ``hi_ij`` -- the 5th/95th percentile of ``q_i - q_j`` over the reference corpus.
* ``w_ij = E[|| N(q) (e_i - e_j)/sqrt(2) ||^2]`` with ``N = I - J^+ J`` the Jacobian null-space
  projector: the fraction of that pair's direction that is self-motion. w=1 means moving along it
  does not move the hand at all, w=0 means it moves the hand and is the task's business, not ours.
  This is what makes the penalty proportional to a pair's influence on posture rather than motion,
  and it is why pairs that would fight the pose goal contribute ~nothing.

The reference is one self-contained artifact, ``posture_ref.npz`` next to this file, baked by
``cutamp/scripts/bake_posture_ref.py``. Override with the ``CUTAMP_POSTURE_REF`` env var.

For the FR3 baked from DROID the table recovers q1-q3 as by far the freest pair (w=0.728, band
[-0.44, +0.56]) with q3-q5 (0.395) and q1-q7 (0.263) next -- i.e. the shoulder null-space
coordinate falls out of the data rather than being assumed.
"""

import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch

DEFAULT_POSTURE_REF = os.environ.get(
    "CUTAMP_POSTURE_REF", str(Path(__file__).resolve().parent / "posture_ref.npz")
)

_CACHE: dict = {}


def load_posture_prior(ref_path: str, device, dtype):
    """Load (and cache) the baked pair table as tensors. Frozen, no grad."""
    key = (ref_path, str(device), str(dtype))
    if key in _CACHE:
        return _CACHE[key]
    blob = np.load(ref_path, allow_pickle=True)
    for k in ("pair_lo", "pair_hi", "pair_weight", "n_joints"):
        if k not in blob:
            raise KeyError(f"{ref_path} has no {k}; rebuild with cutamp/scripts/bake_posture_ref.py")

    def _t(x):
        return torch.as_tensor(np.asarray(x), device=device, dtype=dtype)

    pack = {
        "n_joints": int(blob["n_joints"]),
        "lo": _t(blob["pair_lo"]),          # [J, J]
        "hi": _t(blob["pair_hi"]),          # [J, J]
        "w": _t(blob["pair_weight"]),       # [J, J], zero on the diagonal
        "provenance": str(blob["provenance"]) if "provenance" in blob else "",
    }
    _CACHE[key] = pack
    return pack


def posture_penalty(q: torch.Tensor, pack: dict) -> torch.Tensor:
    """Weighted out-of-band distance summed over joint pairs.

    ``q``: [..., dof] joint positions. Returns [...] -- 0 for a configuration whose every pair
    difference sits inside the human range. Uses the leading ``n_joints`` of ``q``.
    """
    J = pack["n_joints"]
    x = q[..., :J]
    d = x.unsqueeze(-1) - x.unsqueeze(-2)                      # [..., J, J]  d[i,j] = q_i - q_j
    out = (d - pack["hi"]).clamp(min=0.0) + (pack["lo"] - d).clamp(min=0.0)
    # w is zero on the diagonal, and the (i,j)/(j,i) pair is counted twice -- symmetric, so this is
    # a constant factor of 2 on every term and does not change the ranking.
    return (out * pack["w"]).flatten(-2).sum(-1)


def posture_ref_summary(ref_path: Optional[str] = None, top: int = 6) -> str:
    """Human-readable summary of a baked reference, for logging."""
    pack = load_posture_prior(ref_path or DEFAULT_POSTURE_REF, "cpu", torch.float32)
    w = pack["w"].numpy()
    lo, hi = pack["lo"].numpy(), pack["hi"].numpy()
    iu = np.triu_indices(pack["n_joints"], 1)
    order = np.argsort(w[iu])[::-1][:top]
    rows = ", ".join(
        f"q{iu[0][k]+1}-q{iu[1][k]+1} w={w[iu[0][k], iu[1][k]]:.2f} "
        f"[{lo[iu[0][k], iu[1][k]]:+.2f},{hi[iu[0][k], iu[1][k]]:+.2f}]"
        for k in order
    )
    return f"{pack['provenance']} | top pairs: {rows}"
