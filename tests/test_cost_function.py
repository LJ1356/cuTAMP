"""Tests for cutamp.cost_function."""

import os

import pytest
import torch

from cutamp.algorithm import run_cutamp
from cutamp.config import TAMPConfiguration
from cutamp.constraint_checker import ConstraintChecker
from cutamp.cost_reduction import CostReducer
from cutamp.envs.utils import get_env_dir, load_env
from cutamp.scripts.utils import default_constraint_to_mult, default_constraint_to_tol

gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")


def _run_blocks_activation_test(mask: bool) -> int:
    """Run planning on the blocks_activation_dist_test env and return num satisfying particles."""
    env = load_env(os.path.join(get_env_dir(), "blocks_activation_dist_test.yml"))
    config = TAMPConfiguration(
        num_particles=512,
        robot="fr3_robotiq",
        num_opt_steps=500,
        max_loop_dur=20.0,
        enable_visualizer=False,
        rr_spawn=False,
        enable_experiment_logging=False,
        world_activation_distance=0.0,
        mask_initial_movable_world_collision=mask,
    )
    cost_reducer = CostReducer(default_constraint_to_mult.copy())
    constraint_checker = ConstraintChecker(default_constraint_to_tol.copy())
    _, num_satisfying, _ = run_cutamp(env, config, cost_reducer, constraint_checker)
    return num_satisfying


@gpu
def test_movable_to_world_masking():
    """Masking initial movable-to-world collisions should find satisfying particles when blocks
    slightly penetrate the floor (simulating perception noise), while disabling masking should not."""
    assert _run_blocks_activation_test(mask=False) == 0
    assert _run_blocks_activation_test(mask=True) > 0


class _FakeGraspCost:
    """Stand-in for a GraspCost instance; only ``params[1]`` (the grasp parameter name) is read."""

    def __init__(self, obj: str, grasp: str):
        self.params = (obj, grasp)


def _grasp_cost_host(grasp_names: list, **flags):
    """A CostFunction with just the attributes the grasp soft costs read.

    Built without __init__ so the terms can be exercised on CPU, without a TAMPWorld or a solve.
    """
    from cutamp.cost_function import CostFunction

    host = object.__new__(CostFunction)
    host.grasp_costs = [_FakeGraspCost(f"obj{i}", g) for i, g in enumerate(grasp_names)]
    host.grasp_cost_action_names = list(grasp_names)
    host.config = TAMPConfiguration(**flags)
    return host


def test_grasp_center_offset_is_horizontal_distance_from_the_object_origin():
    """grasp_center_offset charges ||xy|| of obj_from_grasp's translation, in meters, ignoring z."""
    host = _grasp_cost_host(["g0"], grasp_center_cost=True)
    mats = torch.eye(4).repeat(3, 1, 1)
    mats[0, :3, 3] = torch.tensor([0.00, 0.00, 0.05])  # dead center; the 5cm of z must not be charged
    mats[1, :3, 3] = torch.tensor([0.03, 0.04, 0.00])  # 5cm off-center horizontally (3-4-5)
    mats[2, :3, 3] = torch.tensor([-0.12, 0.00, 0.02])  # 12cm off, out at an edge
    out = host.grasp_soft_costs({"grasp_to_obj_from_grasp": {"g0": mats}})

    assert out["type"] == "cost" and out["costs"] is host.grasp_costs
    assert set(out["values"]) == {"grasp_center_offset"}  # orientation term stays off when not gated
    offsets = out["values"]["grasp_center_offset"]
    assert offsets.shape == (3, 1)
    assert torch.allclose(offsets.squeeze(1), torch.tensor([0.0, 0.05, 0.12]), atol=1e-6)


def test_grasp_center_offset_covers_every_grasp_in_the_skeleton():
    """One column per grasp parameter, in the order the GraspCosts were collected."""
    host = _grasp_cost_host(["g0", "g1"], grasp_center_cost=True)
    g0, g1 = torch.eye(4).repeat(2, 1, 1), torch.eye(4).repeat(2, 1, 1)
    g0[:, 0, 3] = torch.tensor([0.01, 0.02])
    g1[:, 1, 3] = torch.tensor([0.03, 0.04])
    offsets = host.grasp_soft_costs({"grasp_to_obj_from_grasp": {"g0": g0, "g1": g1}})["values"][
        "grasp_center_offset"
    ]
    assert offsets.shape == (2, 2)
    assert torch.allclose(offsets, torch.tensor([[0.01, 0.03], [0.02, 0.04]]), atol=1e-6)


def test_grasp_soft_costs_emit_nothing_unless_opted_in():
    """The gate matters: CostReducer charges an ABSENT multiplier at weight 1.0, so a value emitted
    without opt-in would silently change every caller's objective."""
    rollout = {"grasp_to_obj_from_grasp": {"g0": torch.eye(4).repeat(2, 1, 1)}}
    assert _grasp_cost_host(["g0"]).grasp_soft_costs(rollout) is None
    assert _grasp_cost_host(["g0"]).grasp_center_costs(rollout) is None
    # ...and a skeleton with no grasps at all emits nothing even when the flag is on.
    assert _grasp_cost_host([], grasp_center_cost=True).grasp_soft_costs(rollout) is None
