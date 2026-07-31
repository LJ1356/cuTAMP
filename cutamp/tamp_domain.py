# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""
Domain file like in PDDL. The task planner can be improved, but it suffices for now.
"""

from typing import Sequence

from cutamp.task_planning import Fluent, Parameter, TAMPOperator, State
from cutamp.task_planning.constraints import (
    CollisionFree,
    DualKinematicConstraint,
    CollisionFreeGrasp,
    CollisionFreeHolding,
    CollisionFreePlacement,
    KinematicConstraint,
    Motion,
    StablePlacement,
    ValidPush,
    ValidPushStick,
)
from cutamp.task_planning.costs import GraspCost, TrajectoryLength

# Types
Conf = "conf"
Traj = "traj"
Pose = "pose"
Grasp = "grasp"

Movable = "movable"
Surface = "surface"
Button = "button"

# Fluents (aka predicates)
At = Fluent("At", [Parameter("q", Conf)])
HandEmpty = Fluent("HandEmpty")
CanMove = Fluent("CanMove")
JustMoved = Fluent("JustMoved")
Holding = Fluent("Holding", [Parameter("obj", Movable)])
HoldingWithGrasp = Fluent("HoldingWithGrasp", [Parameter("obj", Movable), Parameter("grasp", Grasp)])
ButtonPushed = Fluent("ButtonPushed", [Parameter("button", "button")])
PushedWithStick = Fluent("PushedWithStick", [Parameter("button", "button"), Parameter("obj", Movable)])
CanPush = Fluent("CanPush", [Parameter("button", "button")])
IsMovable = Fluent("IsMovable", [Parameter("obj", Movable)])
IsButton = Fluent("IsButton", [Parameter("button", Button)])
IsSurface = Fluent("IsSurface", [Parameter("surface", Surface)])
IsStick = Fluent("IsStick", [Parameter("obj", Movable)])
HasNotPickedUp = Fluent("HasNotPickedUp", [Parameter("obj", Movable)])
# Handover state. The two arms of a handover have FIXED roles -- the giver picks, the taker receives
# -- so the holding hand is identified by the fluent rather than by an arm parameter. That keeps the
# arm out of the type system entirely (no new fabricable type, no per-arm variants of every fluent)
# at the cost of not letting the planner choose which arm picks.
HeldByGiver = Fluent("HeldByGiver", [Parameter("obj", Movable)])
HeldByGiverGrasp = Fluent("HeldByGiverGrasp", [Parameter("obj", Movable), Parameter("grasp", Grasp)])
HeldByTaker = Fluent("HeldByTaker", [Parameter("obj", Movable)])
HeldByTakerGrasp = Fluent("HeldByTakerGrasp", [Parameter("obj", Movable), Parameter("grasp", Grasp)])
On = Fluent("On", [Parameter("obj", Movable), Parameter("surface", Surface)])

all_tamp_fluents = [
    At,
    HandEmpty,
    CanMove,
    JustMoved,
    Holding,
    HoldingWithGrasp,
    ButtonPushed,
    PushedWithStick,
    CanPush,
    IsMovable,
    IsButton,
    IsSurface,
    IsStick,
    HasNotPickedUp,
    HeldByGiver,
    HeldByGiverGrasp,
    HeldByTaker,
    HeldByTakerGrasp,
    On,
]


# Parameters - used for naming in operators, so it's easier to read when debugging
q = Parameter("q", Conf)
q_start = Parameter("q_start", Conf)
q_end = Parameter("q_end", Conf)
traj = Parameter("traj", Traj)

obj = Parameter("obj", Movable)
button = Parameter("button", Button)
surface = Parameter("surface", Surface)

grasp = Parameter("grasp", Grasp)
pose = Parameter("pose", Pose)
placement = Parameter("placement", Pose)

# Per-hand parameters for the lockstep dual-arm operators below. Distinct NAMES are what let one
# fluent be applied to two different objects in the same operator (Fluent.__call__ renames its
# parameter to the one it is called with). `_a` binds to the robot's first arm, `_b` to the second.
obj_a = Parameter("obj_a", Movable)
obj_b = Parameter("obj_b", Movable)
grasp_a = Parameter("grasp_a", Grasp)
grasp_b = Parameter("grasp_b", Grasp)
placement_a = Parameter("placement_a", Pose)
placement_b = Parameter("placement_b", Pose)
surface_a = Parameter("surface_a", Surface)
surface_b = Parameter("surface_b", Surface)

# Handover parameters. `hand_pose` is the object's pose in the WORLD at the moment both hands hold
# it -- a free continuous variable the optimizer solves for, exactly like a placement, except it is
# in mid-air rather than on a surface.
grasp_g = Parameter("grasp_g", Grasp)
grasp_t = Parameter("grasp_t", Grasp)
hand_pose = Parameter("hand_pose", Pose)


# Operators - this is the important part!
MoveFree = TAMPOperator(
    "MoveFree",
    [q_start, traj, q_end],
    preconditions=[At(q_start), HandEmpty(), CanMove()],
    add_effects=[At(q_end), JustMoved()],
    del_effects=[At(q_start), CanMove()],
    constraints=[CollisionFree(q_start, traj, q_end), Motion(q_start, traj, q_end)],
    costs=[TrajectoryLength(q_start, traj, q_end)],
)


MoveHolding = TAMPOperator(
    "MoveHolding",
    [obj, grasp, q_start, traj, q_end],
    preconditions=[
        At(q_start),
        Holding(obj),
        HoldingWithGrasp(obj, grasp),
        CanMove(),
    ],
    add_effects=[At(q_end), JustMoved()],
    del_effects=[At(q_start), CanMove()],
    constraints=[CollisionFreeHolding(obj, grasp, q_start, traj, q_end), Motion(q_start, traj, q_end)],
    costs=[TrajectoryLength(q_start, traj, q_end)],
)


Pick = TAMPOperator(
    "Pick",
    [obj, grasp, q],
    preconditions=[
        At(q),
        HandEmpty(),
        IsMovable(obj),
        JustMoved(),
        HasNotPickedUp(obj),
    ],
    add_effects=[
        Holding(obj),
        HoldingWithGrasp(obj, grasp),
        CanMove(),
    ],
    del_effects=[HandEmpty(), JustMoved(), HasNotPickedUp(obj)],
    constraints=[KinematicConstraint(q, grasp), CollisionFreeGrasp(obj, grasp)],
    costs=[GraspCost(obj, grasp)],
)

Place = TAMPOperator(
    "Place",
    [obj, grasp, placement, surface, q],
    preconditions=[
        At(q),
        Holding(obj),
        HoldingWithGrasp(obj, grasp),
        IsSurface(surface),
        JustMoved(),
    ],
    add_effects=[HandEmpty(), CanMove(), On(obj, surface)],
    del_effects=[
        Holding(obj),
        HoldingWithGrasp(obj, grasp),
        JustMoved(),
    ],
    constraints=[
        KinematicConstraint(q, placement),
        StablePlacement(obj, grasp, placement, surface),
        CollisionFreePlacement(obj, placement, surface),
    ],
    costs=[],
)


Push = TAMPOperator(
    "Push",
    [button, pose, q],
    preconditions=[
        At(q),
        IsButton(button),
        HandEmpty(),
        CanPush(button),
        JustMoved(),
    ],
    add_effects=[ButtonPushed(button), CanMove()],
    del_effects=[JustMoved(), CanPush(button)],  # Note: CFree for Push already encoded in the Move operator
    constraints=[KinematicConstraint(q, pose), ValidPush(button, pose)],
    costs=[],
)


PushStick = TAMPOperator(
    "PushStick",
    [button, obj, grasp, pose, q],
    preconditions=[
        At(q),
        Holding(obj),
        HoldingWithGrasp(obj, grasp),
        IsButton(button),
        IsStick(obj),
        CanPush(button),
        JustMoved(),
    ],
    add_effects=[ButtonPushed(button), CanMove(), PushedWithStick(button, obj)],
    del_effects=[JustMoved(), CanPush(button)],
    # CFree is automatically handled right now within the operator
    constraints=[KinematicConstraint(q, pose), ValidPushStick(button, obj, pose)],
    costs=[],
)

all_tamp_operators = [MoveFree, MoveHolding, Pick, Place, Push, PushStick]


# --------------------------------------------------------------------------- #
# Lockstep dual-arm operators                                                   #
# --------------------------------------------------------------------------- #
# Both arms act at the SAME timestep on ONE shared configuration: PickBoth grasps two objects at one
# configuration, PlaceBoth releases both at one configuration. That lockstep is what lets the hand
# state stay symmetric, so `HandEmpty()` still means "both hands empty" and NO fluent needs an arm
# argument -- every fluent, every existing operator and every goal file is untouched.
#
# These live in their own operator list (`dual_tamp_operators`); a skeleton is built from one list or
# the other, never mixed, so a dual skeleton can never contain a single-arm Pick that would break the
# symmetry.

PickBoth = TAMPOperator(
    "PickBoth",
    [obj_a, grasp_a, obj_b, grasp_b, q],
    preconditions=[
        At(q),
        HandEmpty(),
        IsMovable(obj_a),
        IsMovable(obj_b),
        JustMoved(),
        HasNotPickedUp(obj_a),
        HasNotPickedUp(obj_b),
    ],
    add_effects=[
        Holding(obj_a),
        HoldingWithGrasp(obj_a, grasp_a),
        Holding(obj_b),
        HoldingWithGrasp(obj_b, grasp_b),
        CanMove(),
    ],
    del_effects=[HandEmpty(), JustMoved(), HasNotPickedUp(obj_a), HasNotPickedUp(obj_b)],
    constraints=[
        DualKinematicConstraint(q, grasp_a, grasp_b),
        CollisionFreeGrasp(obj_a, grasp_a),
        CollisionFreeGrasp(obj_b, grasp_b),
    ],
    costs=[GraspCost(obj_a, grasp_a), GraspCost(obj_b, grasp_b)],
    distinct_params=(("obj_a", "obj_b"),),
)

# Exists only so the plan does not ground plain MoveHolding once per held object with identical
# effects; it constrains no end-effector and is skipped by the rollout, like MoveFree.
MoveHoldingBoth = TAMPOperator(
    "MoveHoldingBoth",
    [obj_a, grasp_a, obj_b, grasp_b, q_start, traj, q_end],
    preconditions=[
        At(q_start),
        Holding(obj_a),
        HoldingWithGrasp(obj_a, grasp_a),
        Holding(obj_b),
        HoldingWithGrasp(obj_b, grasp_b),
        CanMove(),
    ],
    add_effects=[At(q_end), JustMoved()],
    del_effects=[At(q_start), CanMove()],
    constraints=[
        CollisionFreeHolding(obj_a, grasp_a, q_start, traj, q_end),
        CollisionFreeHolding(obj_b, grasp_b, q_start, traj, q_end),
        Motion(q_start, traj, q_end),
    ],
    costs=[TrajectoryLength(q_start, traj, q_end)],
    distinct_params=(("obj_a", "obj_b"),),
)

PlaceBoth = TAMPOperator(
    "PlaceBoth",
    [obj_a, grasp_a, placement_a, surface_a, obj_b, grasp_b, placement_b, surface_b, q],
    preconditions=[
        At(q),
        Holding(obj_a),
        HoldingWithGrasp(obj_a, grasp_a),
        Holding(obj_b),
        HoldingWithGrasp(obj_b, grasp_b),
        IsSurface(surface_a),
        IsSurface(surface_b),
        JustMoved(),
    ],
    add_effects=[HandEmpty(), CanMove(), On(obj_a, surface_a), On(obj_b, surface_b)],
    del_effects=[
        Holding(obj_a),
        HoldingWithGrasp(obj_a, grasp_a),
        Holding(obj_b),
        HoldingWithGrasp(obj_b, grasp_b),
        JustMoved(),
    ],
    constraints=[
        DualKinematicConstraint(q, placement_a, placement_b),
        StablePlacement(obj_a, grasp_a, placement_a, surface_a),
        StablePlacement(obj_b, grasp_b, placement_b, surface_b),
        CollisionFreePlacement(obj_a, placement_a, surface_a),
        CollisionFreePlacement(obj_b, placement_b, surface_b),
    ],
    costs=[],
    distinct_params=(("obj_a", "obj_b"),),
)

# MoveFree is reused verbatim: it constrains no end-effector and its Motion / TrajectoryLength terms
# work on whatever width the configuration is.
dual_tamp_operators = [MoveFree, PickBoth, MoveHoldingBoth, PlaceBoth]


# --------------------------------------------------------------------------- #
# Handover                                                                      #
# --------------------------------------------------------------------------- #
# One arm picks an object, both hands meet on it, the second arm carries on. Unlike the lockstep
# operators above this is ASYMMETRIC -- at the pick only the giver acts, at the place only the taker
# -- which is why the rollout tracks WHICH arms each timestep constrains (see `arm_active`) instead
# of assuming every timestep constrains all of them.
#
# The moment of exchange is itself lockstep: `Handover` puts both hands on the same object at one
# configuration, with `DualKinematicConstraint` tying both to the same free object pose.

PickGiver = TAMPOperator(
    "PickGiver",
    [obj, grasp_g, q],
    preconditions=[At(q), HandEmpty(), IsMovable(obj), JustMoved(), HasNotPickedUp(obj)],
    add_effects=[HeldByGiver(obj), HeldByGiverGrasp(obj, grasp_g), CanMove()],
    del_effects=[HandEmpty(), JustMoved(), HasNotPickedUp(obj)],
    constraints=[DualKinematicConstraint(q, grasp_g), CollisionFreeGrasp(obj, grasp_g)],
    costs=[],
)

MoveHoldingGiver = TAMPOperator(
    "MoveHoldingGiver",
    [obj, grasp_g, q_start, traj, q_end],
    preconditions=[At(q_start), HeldByGiver(obj), HeldByGiverGrasp(obj, grasp_g), CanMove()],
    add_effects=[At(q_end), JustMoved()],
    del_effects=[At(q_start), CanMove()],
    constraints=[CollisionFreeHolding(obj, grasp_g, q_start, traj, q_end), Motion(q_start, traj, q_end)],
    costs=[TrajectoryLength(q_start, traj, q_end)],
)

Handover = TAMPOperator(
    "Handover",
    [obj, grasp_g, grasp_t, hand_pose, q],
    preconditions=[At(q), HeldByGiver(obj), HeldByGiverGrasp(obj, grasp_g), JustMoved()],
    add_effects=[HeldByTaker(obj), HeldByTakerGrasp(obj, grasp_t), CanMove()],
    del_effects=[HeldByGiver(obj), HeldByGiverGrasp(obj, grasp_g), JustMoved()],
    # Both hands are constrained to the SAME object pose through their own grasps, which is what
    # makes this an exchange rather than two independent reaches.
    constraints=[DualKinematicConstraint(q, grasp_g, grasp_t), CollisionFreeGrasp(obj, grasp_t)],
    costs=[],
)

MoveHoldingTaker = TAMPOperator(
    "MoveHoldingTaker",
    [obj, grasp_t, q_start, traj, q_end],
    preconditions=[At(q_start), HeldByTaker(obj), HeldByTakerGrasp(obj, grasp_t), CanMove()],
    add_effects=[At(q_end), JustMoved()],
    del_effects=[At(q_start), CanMove()],
    constraints=[CollisionFreeHolding(obj, grasp_t, q_start, traj, q_end), Motion(q_start, traj, q_end)],
    costs=[TrajectoryLength(q_start, traj, q_end)],
)

PlaceTaker = TAMPOperator(
    "PlaceTaker",
    [obj, grasp_t, placement, surface, q],
    preconditions=[At(q), HeldByTaker(obj), HeldByTakerGrasp(obj, grasp_t), IsSurface(surface), JustMoved()],
    add_effects=[HandEmpty(), CanMove(), On(obj, surface)],
    del_effects=[HeldByTaker(obj), HeldByTakerGrasp(obj, grasp_t), JustMoved()],
    constraints=[
        DualKinematicConstraint(q, placement),
        StablePlacement(obj, grasp_t, placement, surface),
        CollisionFreePlacement(obj, placement, surface),
    ],
    costs=[],
)

handover_tamp_operators = [
    MoveFree, PickGiver, MoveHoldingGiver, Handover, MoveHoldingTaker, PlaceTaker
]


def get_initial_state(
    movables: Sequence[str] = (), surfaces: Sequence[str] = (), sticks: Sequence[str] = (), buttons: Sequence[str] = ()
) -> State:
    """Ground the initial state of the TAMP domain."""
    initial_state = {At.ground("q0"), HandEmpty.ground(), CanMove.ground()}
    for movable in movables:
        initial_state.add(IsMovable.ground(movable))
        initial_state.add(HasNotPickedUp.ground(movable))

    for surface in surfaces:
        initial_state.add(IsSurface.ground(surface))

    for stick in sticks:
        initial_state.add(IsStick.ground(stick))
        initial_state.add(IsMovable.ground(stick))
        initial_state.add(HasNotPickedUp.ground(stick))

    for button in buttons:
        initial_state.add(IsButton.ground(button))
        initial_state.add(CanPush.ground(button))

    initial_state = frozenset(initial_state)
    return initial_state
