"""Bimanual YAM: two 6-DOF arms with linear parallel-jaw grippers on a shared base.

Ported from the MolmoAct2 sim-eval MJCF (``allenai/molmoact2`` -> ``sim_eval/robots/bimanual_yam.py``,
assets from the ``TreeePlanter/molmoact2-sim-eval-assets`` HuggingFace dataset). The URDF and the two
cuRobo configs under ``assets/`` are generated from that MJCF by
``droid-sim-evals/tools/yam_mjcf_to_urdf.py``; regenerate rather than hand-editing them.

**One arm at a time.** cuTAMP plans for a single kinematic chain, so each arm gets its own cuRobo
config (``bimanual_yam_left.yml`` / ``bimanual_yam_right.yml``) over the SAME 16-joint URDF. The
config being used sets ``ee_link`` to that arm's grasp frame and ``lock_joints`` on everything else,
which gives cuRobo 6 active DOF while still collision-checking the parked arm and both hands. That
is why the robot identifiers are ``bimanual_yam_left`` / ``bimanual_yam_right`` rather than one
``bimanual_yam``: the arm is part of the embodiment as far as the planner is concerned.

**Frames.** ``<arm>_grasp_frame`` sits on the distal fingertip plane with +z along the approach and
x along the jaw axis -- the same convention as ``grasp_frame`` in the UR5 and FR3 Robotiq URDFs -- so
this robot reuses their ``tool_from_ee`` rotation unchanged.
"""

import logging
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from jaxtyping import Float
from yourdfpy import URDF

from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel
from curobo.geom.types import WorldConfig
from curobo.types.base import TensorDeviceType
from curobo.types.robot import RobotConfig
from curobo.util_file import load_yaml
from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
from cutamp.robots.utils import RerunRobot

_log = logging.getLogger(__name__)

Arm = Literal["left", "right"]
ARMS = ("left", "right")

# Config selecting BOTH arms as one 12-DOF chain (only the four finger joints locked). Its ee_link is
# the left grasp frame and `link_names` tracks both, so a single get_state() returns both hands'
# poses -- which is what lets one configuration be constrained at both hands at once. See
# load_bimanual_yam_dual_container() in cutamp/robots/__init__.py.
DUAL = "dual"
DUAL_JOINT_SLICES = {"left": slice(0, 6), "right": slice(6, 12)}

# Elbow-up posture over a table. Doubles as cuRobo's retract/null-space config and as the pose the
# idle arm is locked at, so it must match ARM_RETRACT in the generator that writes the yml.
bimanual_yam_neutral_joint_positions = (0.0, np.pi / 4, np.pi / 2, 0.0, 0.0, 0.0)

# Finger slide limit, fully open (MJCF actuator ctrlrange; the joint limit is -0.0485). Both fingers
# of both hands are locked here for planning, so cuRobo models the widest the hand ever gets.
YAM_FINGER_OPEN = -0.0475
YAM_FINGER_CLOSED = 0.0

# Clear width between the jaws at full open, in metres (the Robotiq 2F-85 for comparison: 0.085).
YAM_MAX_OPENING = 0.099

# Where along the approach axis the M2T2 contact point sits relative to ``<arm>_grasp_frame``, which
# is authored on the distal fingertip plane. The pads span ee z in [-0.070, 0], i.e. BEHIND the ee
# origin, so a negative value drives the hand further in and seats the object deeper in the jaws.
#
# Zero is what both Franka configs effectively do, but it does not survive contact with this gripper:
# the YAM's jaws are 70 mm deep, so putting the contact on the fingertip plane leaves only the top
# ~20 mm of a 35 mm toy between the pads and the grasp slips. -0.012 drops the fingertip plane to
# roughly the table surface so the pads straddle the toy's full height. It is bounded by the table,
# not by reach: measured on real M2T2 grasps, IK feasibility is flat to -0.012 and then collapses
# (blue toy 95/130 here vs 21/130 at -0.016) as the fingertip spheres start reporting a collision
# with the table top.
YAM_GRASP_DEPTH = -0.012

_ASSETS_DIR = Path(__file__).parent / "assets"


def _check_arm(arm: str, allow_dual: bool = False) -> str:
    allowed = (*ARMS, DUAL) if allow_dual else ARMS
    if arm not in allowed:
        raise ValueError(f"arm must be one of {allowed}, got {arm!r}")
    return arm


def yam_ee_link(arm: Arm) -> str:
    """cuRobo link whose pose an arm's grasps are compared against."""
    return f"{_check_arm(arm)}_grasp_frame"


def yam_attached_link(arm: Arm) -> str:
    """cuRobo link a grasped object is attached to, in the DUAL config (one slot per hand)."""
    return f"{_check_arm(arm)}_attached_object"


def yam_arm_joints(arm: Arm) -> list[str]:
    """The 6 revolute joint names of one arm, in cuRobo/URDF order."""
    return [f"{_check_arm(arm)}_joint{i}" for i in range(1, 7)]


def yam_finger_joints(arm: Arm | None = None) -> list[str]:
    """Finger slide joint names for one arm, or for both hands when ``arm`` is None."""
    arms = ARMS if arm is None else (_check_arm(arm),)
    return [f"{a}_{side}_finger" for a in arms for side in ("left", "right")]


@lru_cache(maxsize=3)
def _bimanual_yam_curobo_cfg_cached(arm: str) -> dict:
    cfg = load_yaml(str(_ASSETS_DIR / f"bimanual_yam_{arm}.yml"))
    # Set some asset paths so cuRobo can load our URDF and meshes
    for key in ("external_asset_path", "external_robot_configs_path"):
        if key not in cfg["robot_cfg"]["kinematics"]:
            cfg["robot_cfg"]["kinematics"][key] = str(_ASSETS_DIR)
    return cfg


def bimanual_yam_curobo_cfg(arm: Arm = "left") -> dict:
    """cuRobo robot config planning with ``arm``; the other arm and both grippers are locked.

    Returns a fresh deep copy each call. Callers mutate it in place -- tiptop's ``get_motion_gen``
    writes ``kinematics.extra_collision_spheres["attached_object"]`` -- so handing out the cached
    dict would let one caller's attachment budget leak into the next.
    """
    return deepcopy(_bimanual_yam_curobo_cfg_cached(_check_arm(arm, allow_dual=True)))


def _yam_cfg_dict(arm: Arm) -> dict:
    return bimanual_yam_curobo_cfg(arm)["robot_cfg"]


def get_bimanual_yam_kinematics_model(arm: str = "left") -> CudaRobotModel:
    """cuRobo kinematics model: 6 active DOF for one arm, or 12 for ``arm="dual"``."""
    robot_cfg = RobotConfig.from_dict(_yam_cfg_dict(arm))
    return CudaRobotModel(robot_cfg.kinematics)


def get_bimanual_yam_ik_solver(
    world_cfg: WorldConfig,
    arm: str = "left",
    num_seeds: int = 24,
    self_collision_opt: bool = False,
    self_collision_check: bool = True,
    use_particle_opt: bool = False,
) -> IKSolver:
    """cuRobo IK solver for one YAM arm.

    More seeds than the 7-DOF arms (24 vs 12): with only 6 DOF there is no null space to slide
    along, so a seed that lands in the wrong elbow branch simply fails instead of being recovered.
    """
    ik_config = IKSolverConfig.load_from_robot_config(
        _yam_cfg_dict(arm),
        world_cfg,
        num_seeds=num_seeds,
        self_collision_opt=self_collision_opt,
        self_collision_check=self_collision_check,
        use_particle_opt=use_particle_opt,
        use_cuda_graph=False,  # matches get_fr3_robotiq_ik_solver: CUDA-graph replay faults on some drivers
    )
    return IKSolver(ik_config)


def yam_tool_from_ee(tensor_args: TensorDeviceType = TensorDeviceType()) -> Float[torch.Tensor, "4 4"]:
    """Transform placing the cuRobo ee frame in cuTAMP's grasp ("tool") frame.

    cuTAMP composes ``world_from_ee = world_from_grasp @ tool_from_ee``. In the grasp frame the
    origin is the contact point on the fingertip plane, +z points back toward the wrist and y is the
    jaw axis -- verified against both existing conventions (the Panda's 0.105 z-offset lands on its
    0.1034 fingertip plane; the Robotiq's pad meshes end at z = -0.0007 in its ``grasp_frame``).

    ``<arm>_grasp_frame`` is authored on the fingertip plane with x along the jaws, identical to the
    Robotiq's ``grasp_frame``, so the rotation is the same rpy = (pi, 0, pi/2). Sanity check on the
    result: it puts the hand's collision spheres at tool y = +/-0.051 spanning z = 0.002..0.120,
    the same shape as the shipped Robotiq set (y = +/-0.062, z = 0.002..0.130).
    """
    import roma

    tool_from_ee = torch.eye(4, device=tensor_args.device)
    rpy = tensor_args.to_device([torch.pi, 0.0, torch.pi / 2])
    tool_from_ee[:3, :3] = roma.euler_to_rotmat("XYZ", rpy)
    tool_from_ee[:3, 3] = tensor_args.to_device([0.0, 0.0, YAM_GRASP_DEPTH])
    return tool_from_ee


@lru_cache(maxsize=2)
def _yam_gripper_spheres_np(arm: Arm) -> Float[np.ndarray, "n 4"]:
    """Hand collision spheres in the grasp ("tool") frame, at full open.

    Derived from the cuRobo config rather than shipped as a ``.pt`` blob (which is what the shared
    Robotiq gripper does) so the two can never drift: the same ``collision_spheres`` the motion
    planner uses are re-expressed in the tool frame here, and regenerating the yml regenerates these.
    Used by cuTAMP particle initialization to reject grasps whose gripper volume hits the object.
    """
    _check_arm(arm)
    kin = _yam_cfg_dict(arm)["kinematics"]
    urdf = URDF.load(
        str(_ASSETS_DIR / kin["urdf_path"]), load_meshes=False, build_collision_scene_graph=False
    )
    urdf.update_cfg(dict.fromkeys(yam_finger_joints(), YAM_FINGER_OPEN))

    base, ee = kin["base_link"], kin["ee_link"]
    base_from_ee = urdf.get_transform(ee, base)
    tool_from_ee = yam_tool_from_ee(TensorDeviceType(device=torch.device("cpu"))).numpy()

    # Only the hand: the arm links behind the wrist are never near a candidate grasp, and including
    # them would reject grasps for touching geometry that is a metre away.
    hand_links = {f"{arm}_link_6", f"{arm}_link_left_finger", f"{arm}_link_right_finger"}
    out = []
    for link, spheres in kin["collision_spheres"].items():
        if link not in hand_links:
            continue
        ee_from_link = np.linalg.inv(base_from_ee) @ urdf.get_transform(link, base)
        for s in spheres:
            p_ee = ee_from_link[:3, :3] @ np.asarray(s["center"], dtype=float) + ee_from_link[:3, 3]
            p_tool = tool_from_ee[:3, :3] @ p_ee + tool_from_ee[:3, 3]
            out.append([*p_tool, float(s["radius"])])
    if not out:
        raise RuntimeError(f"no hand collision spheres found for the {arm} YAM arm")
    return np.asarray(out, dtype=np.float32)


def get_bimanual_yam_gripper_spheres(
    arm: Arm = "left", tensor_args: TensorDeviceType = TensorDeviceType()
) -> Float[torch.Tensor, "num_spheres 4"]:
    """Collision spheres for one YAM hand.

    IMPORTANT: like the other robots these are in the grasp ("tool") frame with z-up, i.e. the origin
    is the fingertip contact point and the hand extends along +z away from the object.
    """
    return tensor_args.to_device(_yam_gripper_spheres_np(arm))


def load_bimanual_yam_rerun(arm: Arm = "left", load_mesh: bool = True) -> RerunRobot:
    kin = _yam_cfg_dict(arm)["kinematics"]
    urdf_path = _ASSETS_DIR / kin["urdf_path"]
    # Mesh paths in the generated URDF are relative to the URDF's own directory, which is yourdfpy's
    # default resolution, so no filename_handler is needed here (unlike the package:// URDFs).
    urdf = URDF.load(str(urdf_path), load_meshes=load_mesh)
    # RerunRobot drives every actuated joint, so pass all 16: this arm at neutral, the other parked
    # at the same posture it is locked to, and all four fingers open.
    q_all = []
    for a in ARMS:
        q_all += list(bimanual_yam_neutral_joint_positions)
    q_all += [YAM_FINGER_OPEN] * 4
    return RerunRobot(f"bimanual_yam_{arm}", urdf, q_neutral=q_all, load_mesh=load_mesh)


if __name__ == "__main__":
    import rerun as rr

    arm: Arm = "left"
    rr.init(f"bimanual_yam_{arm}_test", spawn=True)
    load_bimanual_yam_rerun(arm)

    kin_model = get_bimanual_yam_kinematics_model(arm)
    q = torch.tensor(bimanual_yam_neutral_joint_positions, device="cuda")[None]
    state = kin_model.get_state(q)
    print(f"ee pose: {state.ee_position[0].cpu().numpy()} {state.ee_quaternion[0].cpu().numpy()}")

    spheres = state.link_spheres_tensor[0].cpu()
    rr.log("spheres", rr.Points3D(positions=spheres[:, :3], radii=spheres[:, 3]))

    gripper_spheres = get_bimanual_yam_gripper_spheres(arm).cpu()
    rr.log("gripper_spheres", rr.Points3D(positions=gripper_spheres[:, :3], radii=gripper_spheres[:, 3]))
