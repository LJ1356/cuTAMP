# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from typing import List, Dict, TypedDict

import torch
from jaxtyping import Float

from cutamp.utils.common import Particles, action_6dof_to_mat4x4, action_4dof_to_mat4x4
from cutamp.config import TAMPConfiguration
from cutamp.tamp_domain import (
    Conf,
    Handover,
    MoveFree,
    MoveHolding,
    MoveHoldingBoth,
    MoveHoldingGiver,
    MoveHoldingTaker,
    Pick,
    PickBoth,
    PickGiver,
    Place,
    PlaceBoth,
    PlaceTaker,
    Push,
    PushStick,
)
from cutamp.tamp_world import (
    TAMPWorld,
)
from cutamp.task_planning import PlanSkeleton


def get_conf_parameters(plan_skeleton: PlanSkeleton, ignore_initial: bool) -> List[str]:
    """Get the parameters of the plan skeletons that are of type Conf. Returns a list of unique parameter names."""
    conf_params = []
    for ground_op in plan_skeleton:
        conf_idxs = [idx for idx, param in enumerate(ground_op.operator.parameters) if param.type == Conf]
        op_conf_params = [ground_op.values[idx] for idx in conf_idxs]
        conf_params.extend(op_conf_params)
    conf_params = list(dict.fromkeys(conf_params))  # remove duplicates

    if ignore_initial:
        assert conf_params[0] == "q0", "Expected first configuration to be q0"
        conf_params = conf_params[1:]  # remove the initial configuration
    return conf_params


class Rollout(TypedDict):
    """Dict that stores results of a rollout."""

    num_particles: int
    confs: Float[torch.Tensor, "num_particles *h d"]
    conf_params: List[str]
    robot_spheres: Float[torch.Tensor, "num_particles *h 4"]
    ee_position: Float[torch.Tensor, "num_particles t 3"]
    ee_quaternion: Float[torch.Tensor, "num_particles t 4"]
    world_from_tool_desired: Float[torch.Tensor, "num_particles *h 4 4"]
    world_from_ee_desired: Float[torch.Tensor, "num_particles *h 4 4"]
    gripper_close: List[bool]
    action_params: List[str]
    obj_to_pose: Dict[str, Float[torch.Tensor, "num_particles *h 4 4"]]
    action_to_ts: Dict[str, int]
    action_to_pose_ts: Dict[str, int]
    ts_to_pose_ts: Dict[int, int]
    # --- multi-arm only; empty / absent for a single-arm robot -------------------------------- #
    # A = number of arms. Every timestep constrains ALL arms at once (lockstep operators), so there
    # is never an idle arm and no activity mask is needed.
    arm_names: tuple
    ee_position_arms: Float[torch.Tensor, "num_particles t a 3"]
    ee_quaternion_arms: Float[torch.Tensor, "num_particles t a 4"]
    world_from_tool_desired_arms: Float[torch.Tensor, "num_particles t a 4 4"]
    world_from_ee_desired_arms: Float[torch.Tensor, "num_particles t a 4 4"]
    # Which arms each timestep constrains. Every entry is True for the lockstep operators, but a
    # handover is asymmetric -- at the pick only the giver acts -- so the cost function charges only
    # the ACTIVE (timestep, arm) pairs rather than assuming all of them.
    arm_active: List[List[bool]]
    # Per timestep, in arm order, for the ACTIVE arms only. Lines up 1:1 with the action parameters
    # of that timestep's DualKinematicConstraint.
    gripper_close_arms: List[List[bool]]
    action_params_arms: List[List[str]]


class RolloutFunction:
    """Rollout function that rolls out the plan skeleton. Only supports robot and object kinematics right now."""

    def __init__(self, plan_skeleton: PlanSkeleton, world: TAMPWorld, config: TAMPConfiguration):
        if config.enable_traj:
            raise NotImplementedError("Trajectories are not supported in rollouts yet")
        self.plan_skeleton = plan_skeleton
        self.world = world
        self.config = config
        self.conf_params = get_conf_parameters(plan_skeleton, ignore_initial=True)
        # Non-empty selects the dual-arm path everywhere below.
        self.arms = world.robot_container.arms
        if self.arms:
            self.arm_link_indices = world.robot_container.arm_link_indices
            self.tool_from_ee_arms = torch.stack([a.tool_from_ee for a in self.arms])  # (A, 4, 4)
        self.obj_to_initial_pose = {obj.name: self.world.get_object_pose(obj) for obj in self.world.movables}

        # Grasp to 4x4 matrix function
        if config.grasp_dof == 4:
            self.grasp_to_mat4x4_fn = action_4dof_to_mat4x4
        elif config.grasp_dof == 6:
            self.grasp_to_mat4x4_fn = action_6dof_to_mat4x4
        else:
            raise ValueError(f"Unsupported {config.grasp_dof=}")

        # Assume placements are 4-DOF
        if config.place_dof != 4:
            raise ValueError(f"Unsupported {config.place_dof=}")

        # Flag for first rollout, used to apply a runtime check
        self._is_first_rollout = True

    def __call__(self, particles: Particles) -> Rollout:
        """
        Rollout particles given the plan skeleton through the world. We keep the rollout information minimal to avoid
        unnecessary computations and backward passes.
        """
        num_particles = particles["q0"].shape[0]

        # Forward kinematics, we use .view() as it's faster than rearrange
        with torch.profiler.record_function("rollout::forward_kinematics"):
            confs = torch.stack([particles[conf] for conf in self.conf_params], dim=1)
            confs_flat = confs.view(-1, confs.shape[-1])
            robot_state = self.world.kin_model.get_state(confs_flat)

            # Store ee pose as position + quaternion directly from cuRobo to avoid
            # the Pose -> matrix -> Pose round-trip in kinematic_costs.
            ee_pose = robot_state.ee_pose
            ee_position = ee_pose.position.view(num_particles, confs.shape[1], 3)
            ee_quaternion = ee_pose.quaternion.view(num_particles, confs.shape[1], 4)

            # Robot link spheres for collision checking from cuRobo
            robot_spheres_flat = robot_state.get_link_spheres()
            robot_spheres = robot_spheres_flat.view(num_particles, confs.shape[1], -1, 4)

            # Per-arm end-effector poses. One get_state() call already produced every tracked link's
            # pose, so both hands come from the same forward kinematics -- that is what allows one
            # configuration to be constrained at both hands. NOTE: cuRobo returns VIEWS into buffers
            # it reuses on the next get_state() call, so these are cloned.
            if self.arms:
                idx = self.arm_link_indices
                a = len(self.arms)
                ee_position_arms = robot_state.links_position[:, idx].view(
                    num_particles, confs.shape[1], a, 3).clone()
                ee_quaternion_arms = robot_state.links_quaternion[:, idx].view(
                    num_particles, confs.shape[1], a, 4).clone()

        # Stores the desired actions. In dual-arm mode each entry of `tool_desired_arms` is a
        # (num_particles, A, 4, 4) stack -- one desired tool pose per arm at that timestep.
        tool_desired_arms: List[Float[torch.Tensor, "num_particles a 4 4"]] = []
        gripper_close_arms: List[List[bool]] = []
        action_params_arms: List[List[str]] = []
        arm_active: List[List[bool]] = []

        def emit_arm_slot(per_arm: dict):
            """Record one actionable timestep of a multi-arm rollout.

            ``per_arm`` maps an arm NAME to ``(world_from_tool, gripper_close, action_param)`` and may
            cover only some arms -- an arm absent from it is idle at this timestep and is neither
            given a desired pose nor charged a kinematic cost. Inactive slots are filled with the
            identity purely to keep the stacked tensor rectangular; nothing reads them.
            """
            eye = torch.eye(4, device=confs.device).expand(num_particles, 4, 4)
            poses, closes, names, active = [], [], [], []
            for spec in self.arms:
                entry = per_arm.get(spec.name)
                active.append(entry is not None)
                poses.append(eye if entry is None else entry[0])
                if entry is not None:
                    closes.append(entry[1])
                    names.append(entry[2])
            tool_desired_arms.append(torch.stack(poses, dim=1))
            gripper_close_arms.append(closes)
            action_params_arms.append(names)
            arm_active.append(active)
        world_from_tool_desired = []
        gripper_close: List[bool] = []
        action_params: List[str] = []
        action_to_ts: Dict[str, int] = {}

        # For pose timestamp (pose_ts), we only accumulate the poses if the operator causes a change in the object pose.
        # These dicts are used to map actions and timestamps to their corresponding pose timestamps.
        action_to_pose_ts: Dict[str, int] = {}
        ts_to_pose_ts: Dict[int, int] = {}

        # 4x4 transformation matrices for grasp parameters (if any)
        grasp_to_mat4x4: Dict[str, Float[torch.Tensor, "num_particles 4 4"]] = {}

        def get_grasp_mat4x4(grasp_name_: str) -> Float[torch.Tensor, "num_particles 4 4"]:
            if grasp_name_ not in grasp_to_mat4x4:
                grasp_ = particles[grasp_name_]
                # grasp_ has shape (n, 4, 4) or (n, 4) or (n, 6)
                if grasp_.shape[1:3] == (4, 4):
                    grasp_to_mat4x4[grasp_name_] = grasp_
                else:
                    grasp_to_mat4x4[grasp_name_] = self.grasp_to_mat4x4_fn(grasp_)
            return grasp_to_mat4x4[grasp_name_]

        # Object poses in world frame for every timestep
        obj_to_pose = {
            obj.name: [self.obj_to_initial_pose[obj.name].expand(num_particles, -1, -1)] for obj in self.world.movables
        }

        def current_pose(obj: str) -> Float[torch.Tensor, "num_particles 4 4"]:
            return obj_to_pose[obj][-1]

        # Timestep of all actionable operators, and the corresponding timestep in the obj_to_pose list
        ts, pose_ts = 0, 0

        # Rollout each ground operator in the plan skeleton
        for ground_op in self.plan_skeleton:
            op_name = ground_op.operator.name

            # Skip the Move operators as we don't support trajectories yet
            if op_name in (MoveFree.name, MoveHolding.name, MoveHoldingBoth.name,
                           MoveHoldingGiver.name, MoveHoldingTaker.name):
                continue

            # Pick
            elif op_name == Pick.name:
                obj_name, grasp_name, _ = ground_op.values
                # Grasp is in object frame
                obj_from_grasp = get_grasp_mat4x4(grasp_name)

                # Compute tool pose in world frame given object and grasp pose
                world_from_obj = current_pose(obj_name)
                world_from_grasp = world_from_obj @ obj_from_grasp

                world_from_tool_desired.append(world_from_grasp)
                gripper_close.append(True)  # closing gripper at Pick
                action_params.append(grasp_name)
                action_to_ts[grasp_name] = ts
                action_to_pose_ts[grasp_name] = pose_ts

            # Place
            elif op_name == Place.name:
                obj_name, grasp_name, place_name, _, _ = ground_op.values

                # Place is desired object pose in world frame
                place_4dof = particles[place_name]
                world_from_obj = action_4dof_to_mat4x4(place_4dof)

                # Apply the grasp offset to get the tool frame pose
                obj_from_grasp = get_grasp_mat4x4(grasp_name)
                world_from_tool = world_from_obj @ obj_from_grasp

                # Accumulate poses of all the movable objects as we've moved the object
                for obj in self.world.movables:
                    if obj.name == obj_name:
                        obj_to_pose[obj.name].append(world_from_obj)  # use new desired pose
                    else:
                        obj_to_pose[obj.name].append(current_pose(obj.name))
                pose_ts += 1

                world_from_tool_desired.append(world_from_tool)
                gripper_close.append(False)  # opening gripper at Place
                action_params.append(place_name)
                action_to_ts[place_name] = ts
                action_to_pose_ts[place_name] = pose_ts
                ts_to_pose_ts[ts] = pose_ts


            # PickBoth -- one configuration, both hands grasping a different object
            elif op_name == PickBoth.name:
                obj_a, grasp_a, obj_b, grasp_b, _ = ground_op.values
                slot = {}
                for spec, obj_name, grasp_name in zip(self.arms, (obj_a, obj_b), (grasp_a, grasp_b)):
                    slot[spec.name] = (current_pose(obj_name) @ get_grasp_mat4x4(grasp_name), True, grasp_name)
                    action_to_ts[grasp_name] = ts
                    action_to_pose_ts[grasp_name] = pose_ts
                emit_arm_slot(slot)

            # PlaceBoth -- one configuration, both hands releasing. Both objects move at the SAME
            # pose timestep (unlike two sequential Places, which advance pose_ts once each).
            elif op_name == PlaceBoth.name:
                obj_a, grasp_a, place_a, _, obj_b, grasp_b, place_b, _, _ = ground_op.values
                new_poses, slot = {}, {}
                for spec, obj_name, grasp_name, place_name in zip(
                    self.arms, (obj_a, obj_b), (grasp_a, grasp_b), (place_a, place_b)
                ):
                    world_from_obj = action_4dof_to_mat4x4(particles[place_name])
                    new_poses[obj_name] = world_from_obj
                    slot[spec.name] = (world_from_obj @ get_grasp_mat4x4(grasp_name), False, place_name)
                for obj in self.world.movables:
                    obj_to_pose[obj.name].append(new_poses.get(obj.name, current_pose(obj.name)))
                pose_ts += 1
                for place_name in (place_a, place_b):
                    action_to_ts[place_name] = ts
                    action_to_pose_ts[place_name] = pose_ts
                ts_to_pose_ts[ts] = pose_ts
                emit_arm_slot(slot)

            # PickGiver -- only the GIVER arm acts; the taker is idle and uncharged.
            elif op_name == PickGiver.name:
                obj_name, grasp_name, _ = ground_op.values
                giver = self.arms[0]
                emit_arm_slot({giver.name: (current_pose(obj_name) @ get_grasp_mat4x4(grasp_name),
                                            True, grasp_name)})
                action_to_ts[grasp_name] = ts
                action_to_pose_ts[grasp_name] = pose_ts

            # Handover -- BOTH hands on the same object at one configuration. The object's pose is a
            # free parameter (mid-air, not on any surface); each hand's desired tool pose is that one
            # pose seen through its own grasp, which is what makes this an exchange rather than two
            # independent reaches. The giver releases and the taker closes at this instant.
            elif op_name == Handover.name:
                obj_name, grasp_g_name, grasp_t_name, hand_pose_name, _ = ground_op.values
                world_from_obj = action_4dof_to_mat4x4(particles[hand_pose_name])
                giver, taker = self.arms[0], self.arms[1]
                emit_arm_slot({
                    giver.name: (world_from_obj @ get_grasp_mat4x4(grasp_g_name), False, grasp_g_name),
                    taker.name: (world_from_obj @ get_grasp_mat4x4(grasp_t_name), True, grasp_t_name),
                })
                # The object moves to the handover pose, so this advances the pose timeline exactly
                # like a Place does -- collision checking of the object against the world at this
                # timestep is what keeps the exchange above the table.
                for obj in self.world.movables:
                    obj_to_pose[obj.name].append(
                        world_from_obj if obj.name == obj_name else current_pose(obj.name)
                    )
                pose_ts += 1
                for name in (grasp_g_name, grasp_t_name, hand_pose_name):
                    action_to_ts[name] = ts
                    action_to_pose_ts[name] = pose_ts
                ts_to_pose_ts[ts] = pose_ts

            # PlaceTaker -- only the TAKER arm acts.
            elif op_name == PlaceTaker.name:
                obj_name, grasp_name, place_name, _, _ = ground_op.values
                taker = self.arms[1]
                world_from_obj = action_4dof_to_mat4x4(particles[place_name])
                emit_arm_slot({taker.name: (world_from_obj @ get_grasp_mat4x4(grasp_name),
                                            False, place_name)})
                for obj in self.world.movables:
                    obj_to_pose[obj.name].append(
                        world_from_obj if obj.name == obj_name else current_pose(obj.name)
                    )
                pose_ts += 1
                action_to_ts[place_name] = ts
                action_to_pose_ts[place_name] = pose_ts
                ts_to_pose_ts[ts] = pose_ts

            # Push
            elif op_name == Push.name:
                button_name, pose_name, _ = ground_op.values

                # Push pose is desired tool pose
                push_4dof = particles[pose_name]
                world_from_push = action_4dof_to_mat4x4(push_4dof)

                world_from_tool_desired.append(world_from_push)
                gripper_close.append(True)  # close gripper at Push
                action_params.append(pose_name)
                action_to_ts[pose_name] = ts
                action_to_pose_ts[pose_name] = pose_ts

            # PushStick
            elif op_name == PushStick.name:
                button_name, stick_name, grasp_name, pose_name, _ = ground_op.values

                # Pose is desired stick pose
                stick_4dof = particles[pose_name]
                world_from_stick = action_4dof_to_mat4x4(stick_4dof)

                # Apply the grasp offset to get the tool frame pose
                obj_from_grasp = get_grasp_mat4x4(grasp_name)
                world_from_tool = world_from_stick @ obj_from_grasp

                # Accumulate poses of all the movable objects as we've moved and pushing with stick
                for obj in self.world.movables:
                    if obj.name == stick_name:
                        obj_to_pose[obj.name].append(world_from_stick)
                    else:
                        obj_to_pose[obj.name].append(current_pose(obj.name))
                pose_ts += 1

                world_from_tool_desired.append(world_from_tool)
                gripper_close.append(True)  # close gripper at PushStick (it's still closed)
                action_params.append(pose_name)
                action_to_ts[pose_name] = ts
                action_to_pose_ts[pose_name] = pose_ts

            # Unknown
            else:
                raise ValueError(f"Unsupported operator {op_name}")

            # Increment time step
            ts_to_pose_ts[ts] = pose_ts
            ts += 1

        # Stack and store in rollout
        if self.arms:
            world_from_tool_desired_arms = torch.stack(tool_desired_arms, dim=1)          # (b, T, A, 4, 4)
            world_from_ee_desired_arms = world_from_tool_desired_arms @ self.tool_from_ee_arms
            # The single-arm keys mirror the FIRST arm so every downstream consumer that has not been
            # taught about arms (shape assertions, soft costs, plan-graph export) still sees a
            # well-formed rollout rather than an empty stack.
            first = [next(i for i, on in enumerate(row) if on) for row in arm_active]
            sel = torch.arange(len(first), device=confs.device), torch.tensor(first, device=confs.device)
            world_from_tool_desired = world_from_tool_desired_arms[:, sel[0], sel[1]]
            world_from_ee_desired = world_from_ee_desired_arms[:, sel[0], sel[1]]
            gripper_close = [g[0] for g in gripper_close_arms]
            action_params = [p[0] for p in action_params_arms]
        else:
            world_from_tool_desired = torch.stack(world_from_tool_desired, dim=1)
            world_from_ee_desired = world_from_tool_desired @ self.world.tool_from_ee
            ee_position_arms = ee_quaternion_arms = None
            world_from_tool_desired_arms = world_from_ee_desired_arms = None
            arm_active = []

        # Object poses for each timestep
        obj_to_pose = {k: torch.stack(v, dim=1) for k, v in obj_to_pose.items()}

        # Sanity check
        if self._is_first_rollout:
            assert (
                confs.shape[1]
                == ee_position.shape[1]
                == ee_quaternion.shape[1]
                == world_from_ee_desired.shape[1]
                == world_from_tool_desired.shape[1]
                == ts
            ), "Number of timesteps do not match"
            for obj, poses in obj_to_pose.items():
                assert poses.shape[1] == pose_ts + 1, f"Number of pose timesteps do not match for {obj}"
            self._is_first_rollout = False

        rollout = Rollout(
            num_particles=num_particles,
            confs=confs,
            conf_params=self.conf_params,
            robot_spheres=robot_spheres,
            ee_position=ee_position,
            ee_quaternion=ee_quaternion,
            world_from_tool_desired=world_from_tool_desired,
            world_from_ee_desired=world_from_ee_desired,
            gripper_close=gripper_close,
            action_params=action_params,
            obj_to_pose=obj_to_pose,
            action_to_ts=action_to_ts,
            action_to_pose_ts=action_to_pose_ts,
            ts_to_pose_ts=ts_to_pose_ts,
            arm_names=tuple(a.name for a in self.arms),
            arm_active=arm_active,
            ee_position_arms=ee_position_arms,
            ee_quaternion_arms=ee_quaternion_arms,
            world_from_tool_desired_arms=world_from_tool_desired_arms,
            world_from_ee_desired_arms=world_from_ee_desired_arms,
            gripper_close_arms=gripper_close_arms,
            action_params_arms=action_params_arms,
        )
        return rollout
