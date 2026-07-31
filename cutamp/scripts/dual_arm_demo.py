"""Simultaneous dual-arm pick-and-place: both hands pick, and both place, at the same time.

Two cubes and two trays, one pair on each side of a bimanual YAM. Run it with ``--arm-mode dual`` and
cuTAMP plans FOUR operators::

    MoveFree(q0, traj1, q1)
    PickBoth(cube_a, grasp1, cube_b, grasp2, q1)
    MoveHoldingBoth(cube_a, grasp1, cube_b, grasp2, q1, traj2, q2)
    PlaceBoth(cube_a, grasp1, pose1, tray_a, cube_b, grasp2, pose2, tray_b, q2)

``q1`` is ONE 12-DOF configuration at which the left hand is on cube_a and the right hand is on
cube_b -- not two configurations executed in sequence. Run the same scene with ``--arm-mode single``
and the same goal takes EIGHT operators, because one chain has to do the objects one after another.

What makes the simultaneity work:

* a 12-DOF cuRobo config whose ``link_names`` tracks both grasp frames, so one ``get_state(q)``
  returns both hands' poses and all 154 collision spheres (``bimanual_yam_dual.yml``);
* ``DualKinematicConstraint(q, action_a, action_b)``, whose cost folds the arm axis into the time
  axis and so reuses the existing reducers, weights and tolerances untouched;
* per-arm IK seeding that picks, out of several IK branches per arm, the combination whose arms are
  clear of each other -- each arm's IK is solved with the other one parked, so independently-chosen
  elbow branches routinely intersect.

Reachability is the binding constraint, not the planner: the arms are mounted 0.48 m apart, so if
both targets sit between the shoulders the upper arms genuinely collide (cuTAMP correctly reports
``Motion/self_collision > 0``), and if they sit too far outboard the hands cannot reach. The default
scene here puts each arm's cube and tray on its own side, inside that band.

``--task handover`` runs the other two-handed problem: both hands on the SAME object. The left arm
picks up a 22 cm bar, passes it to the right arm in mid-air, and the right arm places it in a tray the
left arm cannot reach::

    MoveFree(q0, traj1, q1)
    PickGiver(bar, grasp1, q1)
    MoveHoldingGiver(bar, grasp1, q1, traj2, q2)
    Handover(bar, grasp1, grasp2, pose1, q2)
    MoveHoldingTaker(bar, grasp2, q2, traj3, q3)
    PlaceTaker(bar, grasp2, pose2, tray, q3)

The bar is long on purpose. Two YAM grippers are ~10 cm across, so on a 5 cm cube the best mid-air
exchange leaves 1.4 mm of arm clearance; on this bar, held at +/-8 cm, it leaves 127 mm. The two
grasps therefore have to be at opposite ENDS, which the particle initializer arranges directly
(``_slide_grasps_to_end``) because grasps are sampled, not optimized.

Usage::

    pixi run python -m cutamp.scripts.dual_arm_demo                  # parallel, dual (default)
    pixi run python -m cutamp.scripts.dual_arm_demo --arm-mode single
    pixi run python -m cutamp.scripts.dual_arm_demo --task handover
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch

from curobo.geom.types import Cuboid
from cutamp.algorithm import run_cutamp
from cutamp.config import TAMPConfiguration
from cutamp.constraint_checker import ConstraintChecker
from cutamp.cost_reduction import CostReducer
from cutamp.envs import TAMPEnvironment
from cutamp.scripts.utils import default_constraint_to_mult, default_constraint_to_tol
from cutamp.tamp_domain import HandEmpty, On, dual_tamp_operators, handover_tamp_operators

_log = logging.getLogger(__name__)

# The YAM base rests on the table at (0.24, 0, 0.0551); its left arm is mounted at +0.24 in y and its
# right at -0.24. Table top is at z = 0.0451.
TABLE_Z = 0.0451
CUBE = 0.05


# A handover needs an object long enough for each hand to take an end. Measured on this robot: two
# 10 cm YAM grippers converging on a 5 cm cube leave 1.4 mm of arm-to-arm clearance, whereas a 22 cm
# bar grasped near opposite ends leaves 127 mm and EVERY IK-feasible configuration is also arm-clear.
BAR = (0.05, 0.22, 0.05)


def build_handover_env() -> TAMPEnvironment:
    """One bar on the giver's side, one tray on the taker's side, out of the giver's reach."""

    def box(name, dims, pose, color):
        return Cuboid(name=name, dims=dims, pose=[*pose, 1, 0, 0, 0], color=color)

    table = box("table", [0.70, 1.00, 0.03], [0.5486, 0.0221, TABLE_Z - 0.015], [0.8, 0.7, 0.5, 1.0])
    bar = box("bar", list(BAR), [0.40, 0.26, TABLE_Z + BAR[2] / 2], [0.2, 0.6, 0.3, 1.0])
    tray = box("tray", [0.18, 0.18, 0.01], [0.52, -0.26, TABLE_Z + 0.005], [0.9, 0.9, 0.9, 1.0])
    return TAMPEnvironment(
        name="dual_arm_handover",
        movables=[bar],
        statics=[table, tray],
        type_to_objects={"Movable": [bar], "Surface": [tray, table]},
        goal_state=frozenset({On.ground("bar", "tray"), HandEmpty.ground()}),
    )


def build_env(single_arm: bool) -> TAMPEnvironment:
    """Two cubes and two trays. In dual mode one pair sits on each side of the robot; in single-arm
    mode both pairs sit on the left, since the right arm is not part of that embodiment."""
    y_b = 0.10 if single_arm else -0.25
    tray_y_b = 0.05 if single_arm else -0.23

    def box(name, dims, pose, color):
        return Cuboid(name=name, dims=dims, pose=[*pose, 1, 0, 0, 0], color=color)

    table = box("table", [0.70, 1.00, 0.03], [0.5486, 0.0221, TABLE_Z - 0.015], [0.8, 0.7, 0.5, 1.0])
    tray_a = box("tray_a", [0.16, 0.16, 0.01], [0.52, 0.23, TABLE_Z + 0.005], [0.9, 0.9, 0.9, 1.0])
    tray_b = box("tray_b", [0.16, 0.16, 0.01], [0.52, tray_y_b, TABLE_Z + 0.005], [0.9, 0.9, 0.9, 1.0])
    cube_a = box("cube_a", [CUBE] * 3, [0.38, 0.25, TABLE_Z + CUBE / 2], [0.1, 0.3, 0.9, 1.0])
    cube_b = box("cube_b", [CUBE] * 3, [0.38, y_b, TABLE_Z + CUBE / 2], [0.9, 0.3, 0.1, 1.0])

    return TAMPEnvironment(
        name="dual_arm_parallel_pick_place",
        movables=[cube_a, cube_b],
        statics=[table, tray_a, tray_b],
        type_to_objects={"Movable": [cube_a, cube_b], "Surface": [tray_a, tray_b, table]},
        goal_state=frozenset(
            {On.ground("cube_a", "tray_a"), On.ground("cube_b", "tray_b"), HandEmpty.ground()}
        ),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", choices=("parallel", "handover"), default="parallel",
                    help="parallel = both arms pick and place their own object at the same time; "
                         "handover = one arm picks a bar and passes it to the other")
    ap.add_argument("--arm-mode", choices=("dual", "single"), default="dual")
    ap.add_argument("--num-particles", type=int, default=1024)
    ap.add_argument("--num-opt-steps", type=int, default=400)
    ap.add_argument("--save-plan", type=Path, default=None,
                    help="write the refined plan to JSON so it can be replayed in Isaac "
                         "(droid-sim-evals/tools/replay_dual_plan.py)")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger("curobo").setLevel(logging.WARNING)

    handover = args.task == "handover"
    single = args.arm_mode == "single"
    if handover and single:
        raise SystemExit("--task handover needs --arm-mode dual")
    config = TAMPConfiguration(
        robot="bimanual_yam_left" if single else "bimanual_yam_dual",
        arm_mode=args.arm_mode,
        dual_task=args.task,
        num_particles=args.num_particles,
        num_opt_steps=args.num_opt_steps,
        grasp_dof=4,
        curobo_plan=True,   # refine into executable trajectories (solve_curobo_dual)
        max_motion_refine_attempts=8,
        time_dilation_factor=1.0,
        placement_check="obb",
        enable_visualizer=False,
        enable_experiment_logging=False,
        warmup_ik=False,
        warmup_motion_gen=False,
        # Both arm assignments (cube_a->left vs cube_b->left) are separate skeletons, and only one
        # of them keeps each arm on its own side -- the crossed one makes the upper arms intersect,
        # which shows up as Motion/self_collision and is correctly rejected. Sample enough plans to
        # reach both.
        num_initial_plans=6,
    )

    # The default placement tolerances are 0.0, which no sampled placement can hit exactly; tiptop
    # loosens the same two entries for real scenes (see tiptop/planning.py::run_planning).
    tol = default_constraint_to_tol.copy()
    mult = default_constraint_to_mult.copy()
    for surface in (("tray",) if handover else ("tray_a", "tray_b")):
        tol["StablePlacement"][f"{surface}_in_xy"] = 1e-2
        tol["StablePlacement"][f"{surface}_support"] = 1e-2
        mult["StablePlacement"][f"{surface}_support"] = 1.0

    neutral = (0.0, np.pi / 4, np.pi / 2, 0.0, 0.0, 0.0)
    q_init = torch.tensor(neutral if single else neutral * 2, dtype=torch.float32, device="cuda")

    env = build_handover_env() if handover else build_env(single)
    plan, _, failure = run_cutamp(env, config, CostReducer(mult), ConstraintChecker(tol), q_init=q_init)
    if plan is None:
        _log.error(f"No plan: {failure}")
        return

    _log.info(f"Plan has {len(plan)} steps")
    for step in plan:
        _log.info("   " + str({k: v for k, v in step.items() if k in ("type", "action", "arm", "label")}))

    if args.save_plan is not None:
        # Trajectory positions are the robot's full configuration (12 joints for the dual chain,
        # left arm then right); gripper steps carry which hand they actuate.
        steps = []
        for step in plan:
            if step["type"] == "trajectory":
                steps.append({
                    "type": "trajectory",
                    "label": step["label"],
                    "dt": float(step["dt"]),
                    "positions": step["plan"].position.cpu().numpy().tolist(),
                })
            else:
                steps.append({k: step[k] for k in ("type", "action", "arm", "arms", "label") if k in step})
        objects = {o.name: {"pose": list(map(float, o.pose)), "dims": list(map(float, o.dims))}
                   for o in (*env.movables, *env.statics)}
        args.save_plan.parent.mkdir(parents=True, exist_ok=True)
        args.save_plan.write_text(json.dumps(
            {"arm_mode": args.arm_mode, "dual_task": args.task, "robot": config.robot,
             "q_init": q_init.cpu().numpy().tolist(),
             "objects": objects, "steps": steps}, indent=2))
        _log.info(f"Wrote plan to {args.save_plan}")


if __name__ == "__main__":
    main()
