# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from dataclasses import dataclass
from functools import partial
from typing import Sequence

import roma
import torch
from jaxtyping import Float

from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel
from curobo.geom.transform import quaternion_to_matrix
from curobo.types.base import TensorDeviceType
from .franka import (
    franka_curobo_cfg,
    get_franka_kinematics_model,
    get_franka_gripper_spheres,
    franka_neutral_joint_positions,
    load_franka_rerun,
    fr3_franka_neutral_joint_positions,
    get_fr3_franka_kinematics_model,
    get_fr3_franka_gripper_spheres,
    load_fr3_franka_rerun,
)
from .franka_robotiq import (
    load_fr3_robotiq_rerun,
    fr3_robotiq_neutral_joint_positions,
    get_fr3_robotiq_kinematics_model,
    get_fr3_robotiq_gripper_spheres,
    load_panda_robotiq_rerun,
    panda_robotiq_neutral_joint_positions,
    panda_robotiq_curobo_cfg,
    get_panda_robotiq_kinematics_model,
    get_panda_robotiq_gripper_spheres,
    get_panda_robotiq_ik_solver,
)
from .ur5 import load_ur5_rerun, ur5_home, get_ur5_gripper_spheres, get_ur5_ik_solver, get_ur5_kinematics_model
from .bimanual_yam import (
    ARMS as YAM_ARMS,
    DUAL as YAM_DUAL,
    DUAL_JOINT_SLICES as YAM_DUAL_JOINT_SLICES,
    Arm,
    bimanual_yam_curobo_cfg,
    bimanual_yam_neutral_joint_positions,
    get_bimanual_yam_gripper_spheres,
    get_bimanual_yam_ik_solver,
    get_bimanual_yam_kinematics_model,
    load_bimanual_yam_rerun,
    yam_arm_joints,
    yam_attached_link,
    yam_ee_link,
    yam_tool_from_ee,
)
from .utils import RerunRobot


@dataclass(frozen=True)
class ArmSpec:
    """One arm of a multi-arm robot, for containers whose configuration spans several arms.

    ``ee_link`` must be present in the cuRobo config's ``link_names`` so that a single
    ``kin_model.get_state(q)`` returns every arm's pose at once -- that is what allows ONE
    configuration to be constrained at several hands simultaneously.
    """

    name: str                                        # "left" / "right"
    ee_link: str                                     # cuRobo link this arm's grasps are compared to
    joint_slice: slice                               # this arm's columns within a configuration
    tool_from_ee: Float[torch.Tensor, "4 4"]
    gripper_spheres: Float[torch.Tensor, "n 4"]
    attached_link: str = "attached_object"           # cuRobo link this hand's grasped object attaches to
    link_index: int = 0                              # index of ee_link within kin_model.link_names


@dataclass(frozen=True)
class RobotContainer:
    name: str
    kin_model: CudaRobotModel
    joint_limits: Float[torch.Tensor, "2 d"]
    # Note: in tool frame, not end-effector
    gripper_spheres: Float[torch.Tensor, "n 4"]
    # Transformation from tool pose to end-effector (defined in cuRobo config)
    tool_from_ee: Float[torch.Tensor, "4 4"]
    # Per-arm specs for a multi-arm container. EMPTY for every single-arm robot, which is what keeps
    # the single-arm code paths (and the `tool_from_ee` / `gripper_spheres` fields above) untouched.
    arms: tuple = ()

    @property
    def is_multi_arm(self) -> bool:
        return len(self.arms) > 1

    @property
    def arm_link_indices(self) -> Float[torch.Tensor, "a"]:
        """Row indices into ``kin_model.get_state(q).links_position`` for each arm's ee_link."""
        return torch.as_tensor([a.link_index for a in self.arms], dtype=torch.long,
                               device=self.kin_model.tensor_args.device)

    def arm(self, name: str) -> "ArmSpec":
        for spec in self.arms:
            if spec.name == name:
                return spec
        raise KeyError(f"{self.name} has no arm {name!r}; known: {[a.name for a in self.arms]}")


def load_panda_container(tensor_args: TensorDeviceType) -> RobotContainer:
    kin_model = get_franka_kinematics_model()
    joint_limits = kin_model.kinematics_config.joint_limits.position
    assert joint_limits.shape == (2, 7), f"Invalid joint limits shape: {joint_limits.shape}"

    gripper_spheres = get_franka_gripper_spheres(tensor_args)
    tool_from_ee = torch.eye(4, device=tensor_args.device)
    gripper_down_quat = tensor_args.to_device([0.0, 1.0, 0.0, 0.0])
    tool_from_ee[:3, :3] = quaternion_to_matrix(gripper_down_quat[None])[0]
    tool_from_ee[:3, 3] = tensor_args.to_device([0.0, 0.0, 0.105])
    return RobotContainer("panda", kin_model, joint_limits, gripper_spheres, tool_from_ee)


def load_fr3_franka_container(tensor_args: TensorDeviceType) -> RobotContainer:
    kin_model = get_fr3_franka_kinematics_model()
    joint_limits = kin_model.kinematics_config.joint_limits.position
    assert joint_limits.shape == (2, 7), f"Invalid joint limits shape: {joint_limits.shape}"

    gripper_spheres = get_fr3_franka_gripper_spheres(tensor_args)
    tool_from_ee = torch.eye(4, device=tensor_args.device)
    gripper_down_quat = tensor_args.to_device([0.0, 1.0, 0.0, 0.0])
    tool_from_ee[:3, :3] = quaternion_to_matrix(gripper_down_quat[None])[0]
    tool_from_ee[:3, 3] = tensor_args.to_device([0.0, 0.0, 0.105])
    return RobotContainer("fr3_franka", kin_model, joint_limits, gripper_spheres, tool_from_ee)


def load_panda_robotiq_container(tensor_args: TensorDeviceType) -> RobotContainer:
    kin_model = get_panda_robotiq_kinematics_model()
    joint_limits = kin_model.kinematics_config.joint_limits.position
    assert joint_limits.shape == (2, 7), f"Invalid joint limits shape: {joint_limits.shape}"

    gripper_spheres = get_panda_robotiq_gripper_spheres(tensor_args)
    tool_from_ee = torch.eye(4, device=tensor_args.device)
    rpy = tensor_args.to_device([torch.pi, 0, torch.pi / 2])
    tool_from_ee[:3, :3] = roma.euler_to_rotmat("XYZ", rpy)
    tool_from_ee[:3, 3] = tensor_args.to_device([0.0, 0.0, 0.015])
    return RobotContainer("panda_robotiq", kin_model, joint_limits, gripper_spheres, tool_from_ee)


def load_fr3_robotiq_container(tensor_args: TensorDeviceType) -> RobotContainer:
    kin_model = get_fr3_robotiq_kinematics_model()
    joint_limits = kin_model.kinematics_config.joint_limits.position
    assert joint_limits.shape == (2, 7), f"Invalid joint limits shape: {joint_limits.shape}"

    gripper_spheres = get_fr3_robotiq_gripper_spheres(tensor_args)
    tool_from_ee = torch.eye(4, device=tensor_args.device)
    # Should match UR5 which also uses Robotiq
    rpy = tensor_args.to_device([torch.pi, 0, torch.pi / 2])
    tool_from_ee[:3, :3] = roma.euler_to_rotmat("XYZ", rpy)
    # The Robotiq gripper goes down when closing, so we move the tool frame up by 1.5cm
    # Note: this is based on our Robotiq coupling, it may need minor tuning based on your setup.
    tool_from_ee[:3, 3] = tensor_args.to_device([0.0, 0.0, 0.015])
    return RobotContainer("fr3_robotiq", kin_model, joint_limits, gripper_spheres, tool_from_ee)


def load_ur5_container(tensor_args: TensorDeviceType) -> RobotContainer:
    kin_model = get_ur5_kinematics_model()
    joint_limits = kin_model.kinematics_config.joint_limits.position
    assert joint_limits.shape == (2, 6), f"Invalid joint limits shape: {joint_limits.shape}"

    gripper_spheres = get_ur5_gripper_spheres(tensor_args)
    # See screenshot in assets/ur5_home.png to see gripper frame
    tool_from_ee = torch.eye(4, device=tensor_args.device)
    rpy = tensor_args.to_device([torch.pi, 0, torch.pi / 2])
    tool_from_ee[:3, :3] = roma.euler_to_rotmat("XYZ", rpy)
    # The Robotiq gripper goes down when closing, so we move the tool frame up by 1cm
    tool_from_ee[:3, 3] = tensor_args.to_device([0.0, 0.0, 0.01])
    return RobotContainer("ur5", kin_model, joint_limits, gripper_spheres, tool_from_ee)


def load_bimanual_yam_dual_container(tensor_args: TensorDeviceType) -> RobotContainer:
    """Container driving BOTH YAM arms as one 12-DOF chain, for simultaneous dual-arm plans.

    The two arms share their gripper geometry and their tool frame (the hands are identical, just
    mirrored in placement), so the per-arm specs differ only in which cuRobo link the arm's grasps
    are compared against and which 6 columns of a configuration belong to it. ``tool_from_ee`` and
    ``gripper_spheres`` at the top level mirror the LEFT arm so that any code reaching for the
    single-arm fields still gets something meaningful.
    """
    kin_model = get_bimanual_yam_kinematics_model(YAM_DUAL)
    joint_limits = kin_model.kinematics_config.joint_limits.position
    assert joint_limits.shape == (2, 12), f"Invalid joint limits shape: {joint_limits.shape}"

    tool_from_ee = yam_tool_from_ee(tensor_args)
    link_names = list(kin_model.link_names)
    joint_names = list(kin_model.joint_names)
    arms = tuple(
        ArmSpec(
            name=name,
            ee_link=yam_ee_link(name),
            joint_slice=YAM_DUAL_JOINT_SLICES[name],
            tool_from_ee=tool_from_ee,
            gripper_spheres=get_bimanual_yam_gripper_spheres(name, tensor_args),
            attached_link=yam_attached_link(name),
            # Looked up, never assumed from declaration order: this index selects the arm's row out
            # of the stacked link poses, so getting it wrong silently controls the wrong hand.
            link_index=link_names.index(yam_ee_link(name)),
        )
        for name in YAM_ARMS
    )
    for spec in arms:
        assert joint_names[spec.joint_slice] == yam_arm_joints(spec.name), (
            f"{spec.name} joint slice {spec.joint_slice} maps to {joint_names[spec.joint_slice]}")
        assert link_names[spec.link_index] == spec.ee_link
        # `link_names` is only the TRACKED links; extra_links live in the kinematics link map.
        assert spec.attached_link in kin_model.kinematics_config.link_name_to_idx_map, (
            f"{spec.attached_link} missing from the dual cuRobo config's extra_links")
    return RobotContainer(
        "bimanual_yam_dual", kin_model, joint_limits, arms[0].gripper_spheres, tool_from_ee, arms=arms
    )


def load_bimanual_yam_container(arm: Arm, tensor_args: TensorDeviceType) -> RobotContainer:
    """Container for one YAM arm; the other arm and both grippers are locked (see bimanual_yam.py)."""
    kin_model = get_bimanual_yam_kinematics_model(arm)
    joint_limits = kin_model.kinematics_config.joint_limits.position
    assert joint_limits.shape == (2, 6), f"Invalid joint limits shape: {joint_limits.shape}"

    gripper_spheres = get_bimanual_yam_gripper_spheres(arm, tensor_args)
    # Defined next to the gripper spheres (which are expressed in the frame it defines) rather than
    # inline here, so the two cannot drift apart.
    tool_from_ee = yam_tool_from_ee(tensor_args)
    return RobotContainer(f"bimanual_yam_{arm}", kin_model, joint_limits, gripper_spheres, tool_from_ee)


robot_to_fns = {
    "panda": {
        "rerun": load_franka_rerun,
        "q_home": franka_neutral_joint_positions[:7],  # exclude gripper joint
        "container": load_panda_container,
    },
    "fr3_franka": {
        "rerun": load_fr3_franka_rerun,
        "q_home": fr3_franka_neutral_joint_positions,
        "container": load_fr3_franka_container,
    },
    "panda_robotiq": {
        "rerun": load_panda_robotiq_rerun,
        "q_home": panda_robotiq_neutral_joint_positions,
        "container": load_panda_robotiq_container,
    },
    "fr3_robotiq": {
        "rerun": load_fr3_robotiq_rerun,
        "q_home": fr3_robotiq_neutral_joint_positions,  # include gripper joint
        "container": load_fr3_robotiq_container,
    },
    "ur5": {
        "rerun": load_ur5_rerun,
        "q_home": ur5_home[:6],
        "container": load_ur5_container,
    },
    # The bimanual YAM registers once per arm: cuTAMP plans a single chain, so which arm is active
    # is baked into the cuRobo config (ee_link + lock_joints) and therefore into the robot id.
    **{
        f"bimanual_yam_{arm}": {
            "rerun": partial(load_bimanual_yam_rerun, arm),
            "q_home": bimanual_yam_neutral_joint_positions,
            "container": partial(load_bimanual_yam_container, arm),
        }
        for arm in YAM_ARMS
    },
    # Both arms at once, as a single 12-DOF chain. See load_bimanual_yam_dual_container.
    "bimanual_yam_dual": {
        "rerun": partial(load_bimanual_yam_rerun, "left"),
        "q_home": tuple(bimanual_yam_neutral_joint_positions) * 2,
        "container": load_bimanual_yam_dual_container,
    },
}


def load_rerun_robot(robot: str, load_mesh: bool = True) -> RerunRobot:
    if robot not in robot_to_fns:
        raise ValueError(f"Unknown robot: {robot}. Supported robots: {list(robot_to_fns.keys())}")
    rerun_fn = robot_to_fns[robot]["rerun"]
    rerun_robot = rerun_fn(load_mesh)
    if not isinstance(rerun_robot, RerunRobot):
        raise TypeError(f"Expected RerunRobot, got {type(rerun_robot)}")
    return rerun_robot


def get_q_home(robot: str) -> Sequence[float]:
    """Get the home joint positions for the specified robot."""
    if robot not in robot_to_fns:
        raise ValueError(f"Unknown robot: {robot}. Supported robots: {list(robot_to_fns.keys())}")
    q_home = robot_to_fns[robot]["q_home"]
    return q_home


def load_robot_container(robot: str, tensor_args: TensorDeviceType) -> RobotContainer:
    """Load robot container which contains many helper classes and variables."""
    if robot not in robot_to_fns:
        raise ValueError(f"Unknown robot: {robot}. Supported robots: {list(robot_to_fns.keys())}")
    container_fn = robot_to_fns[robot]["container"]
    container = container_fn(tensor_args)
    if not isinstance(container, RobotContainer):
        raise TypeError(f"Expected RobotContainer, got {type(container)}")
    return container
