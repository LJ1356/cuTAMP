"""Tests for planning with pick-only (Holding) goals."""

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


@gpu
def test_pick_only_goal_finds_plan():
    """Planning with a Holding-only goal should produce satisfying particles."""
    env = load_env(os.path.join(get_env_dir(), "pick_block.yml"))
    config = TAMPConfiguration(
        num_particles=512,
        robot="fr3_robotiq",
        num_opt_steps=500,
        max_loop_dur=20.0,
        enable_visualizer=False,
        rr_spawn=False,
        enable_experiment_logging=False,
    )
    cost_reducer = CostReducer(default_constraint_to_mult.copy())
    constraint_checker = ConstraintChecker(default_constraint_to_tol.copy())
    _, num_satisfying, failure_reason = run_cutamp(env, config, cost_reducer, constraint_checker)
    assert failure_reason is None
    assert num_satisfying > 0


@gpu
@pytest.mark.parametrize("weight", [0.0, 30.0])
def test_grasp_center_cost_runs_and_shapes_the_grasp(weight):
    """Planning still succeeds end to end with the off-center grasp soft cost enabled.

    The term's arithmetic and its gating are asserted directly in
    test_cost_function.py::test_grasp_center_offset*; this covers the wiring -- config gate, rollout
    key, reducer multiplier -- on a real solve.

    weight=0.0 is the baseline (gate off, cost never emitted -- the reducer would otherwise charge an
    absent multiplier at weight 1.0). weight=30.0 opts in via both the gate and the multiplier, the way
    tiptop's `grasp_center_weight` YAML knob does.
    """
    env = load_env(os.path.join(get_env_dir(), "pick_block.yml"))
    config = TAMPConfiguration(
        num_particles=512,
        robot="fr3_robotiq",
        num_opt_steps=500,
        max_loop_dur=20.0,
        enable_visualizer=False,
        rr_spawn=False,
        enable_experiment_logging=False,
        grasp_center_cost=bool(weight),
    )
    mults = default_constraint_to_mult.copy()
    if weight:
        mults["GraspCost"] = {"grasp_center_offset": weight}
    cost_reducer = CostReducer(mults)
    constraint_checker = ConstraintChecker(default_constraint_to_tol.copy())
    plan, num_satisfying, failure_reason = run_cutamp(env, config, cost_reducer, constraint_checker)
    assert failure_reason is None
    assert num_satisfying > 0
