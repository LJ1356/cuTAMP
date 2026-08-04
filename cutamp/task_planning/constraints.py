# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from cutamp.task_planning import Constraint


class Collision(Constraint):
    """This constraint is used more for typing purposes."""

    def __init__(self):
        super().__init__()


class CollisionFree(Constraint):
    def __init__(self, *params):
        super().__init__(*params)


class CollisionFreeGrasp(Constraint):
    def __init__(self, obj, grasp):
        super().__init__(obj, grasp)


class CollisionFreeHolding(Constraint):
    def __init__(self, obj, grasp, *params):
        super().__init__(obj, grasp, *params)


class CollisionFreePlacement(Constraint):
    def __init__(self, obj, placement, surface):
        super().__init__(obj, placement, surface)


class KinematicConstraint(Constraint):
    def __init__(self, conf, target_pose):
        super().__init__(conf, target_pose)


class DualKinematicConstraint(Constraint):
    """One configuration of a multi-arm robot, constrained at one or more of its hands at once.

    Deliberately a distinct class from :class:`KinematicConstraint` rather than several instances of
    it: the cost function zips its kinematic constraints 1:1 against the rollout's timesteps, and two
    KinematicConstraints sharing a conf would break that invariant for every single-arm skeleton too.

    The NUMBER of actions is how many hands this timestep constrains -- two for a lockstep operator
    like PickBoth, one for an asymmetric one like PickGiver, where only the giver acts. WHICH hands
    those are comes from the rollout (``arm_active``), because it depends on the operator's semantics
    rather than on anything visible here.
    """

    def __init__(self, conf, *actions):
        if not actions:
            raise ValueError("DualKinematicConstraint needs at least one action parameter")
        super().__init__(conf, *actions)


class Motion(Constraint):
    def __init__(self, q_start, traj, q_end):
        super().__init__(q_start, traj, q_end)


class StablePlacement(Constraint):
    def __init__(self, obj, grasp, placement, surface):
        super().__init__(obj, grasp, placement, surface)


class ValidPush(Constraint):
    def __init__(self, button, push_pose):
        super().__init__(button, push_pose)


class ValidPushStick(Constraint):
    def __init__(self, button, stick, push_pose):
        super().__init__(button, stick, push_pose)
