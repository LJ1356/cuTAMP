"""
Graph representation of a TAMP plan.

A cuTAMP plan skeleton is, structurally, a **bipartite factor graph**: the continuous
decision variables (configurations, poses, grasps, trajectories) and the physical objects
(movables, surfaces, buttons) are the *variable* nodes, while the constraints and costs that
couple them are the *factor* nodes. The task-level operators own those factors and impose a
temporal ordering. This module materializes that structure into a serializable graph so it can
be inspected, visualized, or analyzed offline.

The graph is emitted as a JSON node-link document (see :func:`build_plan_graph`) and,
optionally, as a Graphviz DOT file for visualization. No third-party graph library is required;
:func:`to_networkx` is provided as a convenience if ``networkx`` happens to be installed.

Node kinds
----------
- ``operator``   : one per ground operator in the skeleton (ordered by plan step).
- ``variable``   : one per distinct parameter value. ``var_class`` is either ``decision``
                   (conf/pose/grasp/traj — optimized) or ``object`` (movable/surface/button).
- ``constraint`` : one per ground constraint (a hard factor that must be satisfied).
- ``cost``       : one per ground cost (a soft factor that is minimized).

Edge relations (``rel``)
------------------------
- ``precedes``         : operator -> next operator (plan order).
- ``uses``             : operator -> variable it is parameterized by.
- ``owns_constraint``  : operator -> constraint it introduces.
- ``owns_cost``        : operator -> cost it introduces.
- ``constrains``       : constraint -> variable in its scope.
- ``affects``          : cost -> variable in its scope.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

_log = logging.getLogger(__name__)

FORMAT_VERSION = "cutamp-plan-graph/v1"

# Parameter type -> variable class. Types come from cutamp.tamp_domain (Conf, Traj, Pose, Grasp,
# Movable, Surface, Button). Decision variables are the ones cuTAMP optimizes; object variables
# are fixed scene entities referenced by name.
_DECISION_TYPES = {"conf", "traj", "pose", "grasp"}
_OBJECT_TYPES = {"movable", "surface", "button"}

# Maps a specific constraint/cost class name to the key it is reported under in the cost dict
# produced by cutamp.cost_function.CostFunction. Several collision-free constraint classes all
# funnel into the single "Collision" bucket. ``None`` means the factor has no numeric report.
_CTYPE_TO_COST_KEY = {
    "CollisionFree": "Collision",
    "CollisionFreeGrasp": "Collision",
    "CollisionFreeHolding": "Collision",
    "CollisionFreePlacement": "Collision",
    "Collision": "Collision",
    "KinematicConstraint": "KinematicConstraint",
    "Motion": "Motion",
    "StablePlacement": "StablePlacement",
    "ValidPush": "ValidPush",
    "ValidPushStick": "ValidPush",
    "TrajectoryLength": "TrajectoryLength",
}


# ---------------------------------------------------------------------------
# Human-readable documentation for the TAMP domain (operators / constraints / costs).
# Embedded into the interactive HTML so nodes are self-describing. Kept in sync with
# cutamp/tamp_domain.py and cutamp/task_planning/constraints.py.
# ---------------------------------------------------------------------------
OPERATOR_DOCS = {
    "MoveFree": {
        "summary": "Move the arm through free space with an empty gripper.",
        "does": "Transitions the robot from configuration q_start to q_end along a collision-free "
        "path while the hand is empty. Requires HandEmpty and being At(q_start); results in At(q_end). "
        "Used to travel to a grasp or between placements.",
        "params": {
            "q_start": "Joint configuration the motion starts from (the arm's current pose).",
            "traj": "The free-space path (waypoint sequence) linking q_start to q_end; checked for "
            "collisions and joint-limit / self-collision validity.",
            "q_end": "Joint configuration the motion ends at (the next Pick's grasp config).",
        },
    },
    "MoveHolding": {
        "summary": "Move the arm while holding an object.",
        "does": "Carries obj (held with grasp) from q_start to q_end. Like MoveFree, but the grasped "
        "object's geometry is included in collision checking.",
        "params": {
            "obj": "The object currently grasped and carried during the motion.",
            "grasp": "The grasp transform (gripper pose relative to obj) fixing where obj sits in the "
            "hand; determines the carried object's swept volume.",
            "q_start": "Start joint configuration (just after the Pick).",
            "traj": "The carrying path, collision-checked for both the robot and the held object.",
            "q_end": "End joint configuration (where the following Place happens).",
        },
    },
    "Pick": {
        "summary": "Grasp an object.",
        "does": "At configuration q, closes the gripper on obj using grasp. Requires HandEmpty, "
        "IsMovable(obj), and HasNotPickedUp(obj); results in Holding(obj) with that grasp.",
        "params": {
            "obj": "The movable object to grasp.",
            "grasp": "The grasp pose — gripper pose relative to obj (4-DOF top-down or 6-DOF). A decision "
            "variable the optimizer positions so the grasp is reachable and collision-free.",
            "q": "The arm joint configuration whose forward kinematics places the gripper exactly at the "
            "grasp pose (tied to grasp by KinematicConstraint).",
        },
    },
    "Place": {
        "summary": "Place a held object onto a surface.",
        "does": "At configuration q, releases obj (held with grasp) at world pose placement, resting on "
        "surface. Results in HandEmpty and On(obj, surface).",
        "params": {
            "obj": "The object being placed.",
            "grasp": "The grasp obj is currently held with; links the object's placement pose to the "
            "required gripper/arm configuration.",
            "placement": "Target world pose of obj after release — a decision variable placed within/on "
            "the surface.",
            "surface": "The surface obj must end up resting on (its footprint bounds the valid region).",
            "q": "The arm joint configuration that achieves the placement pose (tied by KinematicConstraint).",
        },
    },
    "Push": {
        "summary": "Push a button with the empty gripper.",
        "does": "At configuration q, moves the tool to pose to actuate button. Results in ButtonPushed(button).",
        "params": {
            "button": "The button to actuate.",
            "pose": "The tool/end-effector pose used to contact the button — a decision variable.",
            "q": "Arm configuration achieving pose (tied by KinematicConstraint).",
        },
    },
    "PushStick": {
        "summary": "Push a button using a held stick.",
        "does": "At configuration q, uses stick obj (held with grasp) to reach and actuate button at pose. "
        "For buttons out of direct reach.",
        "params": {
            "button": "The button to actuate.",
            "obj": "The stick tool being held.",
            "grasp": "The grasp on the stick.",
            "pose": "The pushing pose — a decision variable.",
            "q": "Arm configuration achieving pose.",
        },
    },
}

# Constraint/cost docs. ``params`` is ordered to match the positional arguments of the constraint
# class in cutamp/task_planning/constraints.py, so they can be zipped with a node's concrete params.
CONSTRAINT_DOCS = {
    "KinematicConstraint": {
        "summary": "Forward-kinematics reachability: FK(conf) must equal the target pose.",
        "detail": "Penalizes the position and orientation error between the end-effector pose produced "
        "by the arm configuration and the desired grasp/placement pose. Zero error ⇒ the arm reaches it.",
        "params": [
            ["conf", "The arm joint configuration whose forward kinematics is evaluated."],
            ["target_pose", "The desired end-effector/tool pose to reach — a grasp pose (Pick) or a "
             "placement pose (Place)."],
        ],
    },
    "Motion": {
        "summary": "The trajectory is a physically valid robot motion.",
        "detail": "Requires every configuration along the motion to stay within joint limits and be "
        "self-collision-free.",
        "params": [
            ["q_start", "Configuration the motion starts from."],
            ["traj", "The trajectory whose configurations are checked for joint-limit and self-collision "
             "violations."],
            ["q_end", "Configuration the motion ends at."],
        ],
    },
    "CollisionFree": {
        "summary": "Free-space motion is collision-free (robot vs world).",
        "detail": "The empty-gripper motion must not collide the robot with static world geometry (or "
        "movable objects) anywhere along the path.",
        "params": [
            ["q_start", "Start configuration of the motion."],
            ["traj", "Path checked for robot–world collisions."],
            ["q_end", "End configuration of the motion."],
        ],
    },
    "CollisionFreeGrasp": {
        "summary": "The grasp itself is collision-free.",
        "detail": "Grasping obj with grasp must not drive the gripper into obj or nearby geometry beyond "
        "the intended contact.",
        "params": [
            ["obj", "The object being grasped."],
            ["grasp", "The candidate grasp pose, evaluated for collision of the gripper against the scene."],
        ],
    },
    "CollisionFreeHolding": {
        "summary": "Carrying motion is collision-free for robot + held object.",
        "detail": "While moving holding obj with grasp, neither the robot nor the grasped object may "
        "collide with the world or other objects along traj.",
        "params": [
            ["obj", "The carried object (its swept geometry is collision-checked)."],
            ["grasp", "Grasp fixing obj's pose in the hand."],
            ["q_start", "Start configuration."],
            ["traj", "Carrying path checked for collision."],
            ["q_end", "End configuration."],
        ],
    },
    "CollisionFreePlacement": {
        "summary": "The placement is collision-free.",
        "detail": "With obj at placement on surface, obj must not overlap other movable objects or world "
        "geometry (resting contact with the surface excepted).",
        "params": [
            ["obj", "The object being placed."],
            ["placement", "The target pose of obj, evaluated for collision against the scene."],
            ["surface", "The surface being placed on (resting contact allowed)."],
        ],
    },
    "StablePlacement": {
        "summary": "The object rests stably on the surface.",
        "detail": "Two conditions: (1) the object's support spheres in xy lie within the surface footprint, "
        "and (2) the object's underside meets the surface height (z) — so it is supported, not floating or "
        "hanging off the edge.",
        "params": [
            ["obj", "The object being placed."],
            ["grasp", "The grasp obj is held with (relates the held geometry to the placement)."],
            ["placement", "The candidate placement pose whose stability is scored."],
            ["surface", "The surface providing support; its footprint/height define the stable region."],
        ],
    },
    "ValidPush": {
        "summary": "The push pose validly actuates the button.",
        "detail": "The tool contact point must fall within the button's actuation region (a small box just "
        "above the button).",
        "params": [
            ["button", "The button to actuate."],
            ["push_pose", "The tool pose whose contact point is checked against the button region."],
        ],
    },
    "ValidPushStick": {
        "summary": "Pushing the button with the stick is valid.",
        "detail": "The held stick's geometry must reach into the button's actuation region at the push pose.",
        "params": [
            ["button", "The button to actuate."],
            ["stick", "The held stick tool used to reach the button."],
            ["push_pose", "The push pose whose resulting stick contact is checked against the button region."],
        ],
    },
    "Collision": {
        "summary": "Aggregate collision marker (typing only).",
        "detail": "Placeholder constraint type; collision costs are computed globally from world geometry "
        "rather than per instance.",
        "params": [],
    },
}

COST_DOCS = {
    "TrajectoryLength": {
        "summary": "Soft cost: joint-space length of the motion.",
        "detail": "Sum of configuration-to-configuration distances along the trajectory; minimized to "
        "prefer shorter, smoother motions. Does not gate feasibility.",
        "params": [
            ["q_start", "Start configuration."],
            ["traj", "The path whose joint-space length is measured."],
            ["q_end", "End configuration."],
        ],
    },
    "GraspCost": {
        "summary": "Soft cost scoring grasp quality.",
        "detail": "Scores how preferred the grasp on obj is. Note: in this fork it is informational — the "
        "differentiable cost function warns that GraspCost is not currently evaluated.",
        "params": [
            ["obj", "The object being grasped."],
            ["grasp", "The grasp pose being scored."],
        ],
    },
}

# Which constraints/costs can be removed, to study how the TAMP data distribution shifts.
#   free       : cleanly removable — the cost function guards on its presence; removing relaxes that
#                requirement and MOVES the sampled distribution.
#   noop       : symbolically removable but with NO numeric effect (e.g. collision is computed from
#                world geometry regardless); relax via tolerance / activation-distance config instead.
#   structural : cannot be removed per instance — the cost function asserts one per rollout variable,
#                so removing it breaks rollout validation.
# (See the earlier removability analysis; kept in sync with cutamp/cost_function.py.)
CONSTRAINT_REMOVABILITY = {
    "StablePlacement": {"tier": "free", "label": "removable",
        "effect": "Placed objects no longer need to rest within/on the surface. Placement poses spread "
        "across and beyond the surface (and may float or overhang), widening the placement (x, y, z) "
        "distribution toward physically unstable configurations."},
    "ValidPush": {"tier": "free", "label": "removable",
        "effect": "The push pose no longer has to reach the button's actuation region, so sampled push "
        "poses spread out instead of concentrating at the button."},
    "ValidPushStick": {"tier": "free", "label": "removable",
        "effect": "The held stick no longer has to reach the button region, spreading the stick push poses."},
    "CollisionFree": {"tier": "noop", "label": "no-op to remove",
        "effect": "Removing this symbolic constraint does NOT disable collision — robot–world collision is "
        "computed globally from geometry. To let trajectories pass through obstacles, raise the collision "
        "tolerance / world activation distance instead."},
    "CollisionFreeGrasp": {"tier": "noop", "label": "no-op to remove",
        "effect": "Grasp collision is computed from geometry regardless of this constraint. Relax via the "
        "gripper activation distance / tolerance, not by removing it."},
    "CollisionFreeHolding": {"tier": "noop", "label": "no-op to remove",
        "effect": "Carrying collision is computed from geometry regardless. Relax via tolerance / activation "
        "distance, not removal."},
    "CollisionFreePlacement": {"tier": "noop", "label": "no-op to remove",
        "effect": "Placement collision is computed from geometry regardless. Relax via tolerance, not removal."},
    "Collision": {"tier": "noop", "label": "no-op to remove",
        "effect": "Typing marker; collision costs come from world geometry, not this node."},
    "KinematicConstraint": {"tier": "structural", "label": "structural — keep",
        "effect": "Cannot be removed: the cost function asserts exactly one kinematic constraint per "
        "(configuration, target pose). It ties each arm config to its grasp/placement pose; without it the "
        "rollout's variables are undefined and validation fails."},
    "Motion": {"tier": "structural", "label": "structural — keep",
        "effect": "Cannot be removed per instance: the cost function requires every rollout configuration to "
        "be covered by a Motion constraint (joint limits + self-collision). Removing it fails validation."},
    "TrajectoryLength": {"tier": "structural", "label": "structural — keep",
        "effect": "Soft cost, but removing it breaks rollout validation (its confs must equal the rollout's). "
        "To stop shaping toward short paths, set its multiplier to 0 rather than removing it."},
    "GraspCost": {"tier": "noop", "label": "no-op to remove",
        "effect": "Already not evaluated by the differentiable cost function in this fork (a warning is "
        "emitted), so removing it changes nothing."},
}

PLAN_DOCS = {
    "operators": OPERATOR_DOCS,
    "constraints": CONSTRAINT_DOCS,
    "costs": COST_DOCS,
    "removability": CONSTRAINT_REMOVABILITY,
}


def _var_class(var_type: Optional[str]) -> str:
    if var_type in _DECISION_TYPES:
        return "decision"
    if var_type in _OBJECT_TYPES:
        return "object"
    return "unknown"


def _param_type_map(plan_skeleton: Sequence[Any]) -> Dict[str, str]:
    """Map each parameter *value* name (e.g. ``q1``, ``grasp2``, ``toy0``) to its declared type."""
    value_to_type: Dict[str, str] = {}
    for ground_op in plan_skeleton:
        types = [p.type for p in ground_op.operator.parameters]
        for value, vtype in zip(ground_op.values, types):
            # Values are consistently typed across the skeleton; keep the first binding.
            value_to_type.setdefault(value, vtype)
    return value_to_type


def _scalar(value: Any) -> Optional[float]:
    """Best-effort conversion of a (possibly torch) scalar to a Python float."""
    try:
        if hasattr(value, "item"):
            return float(value.item())
        return float(value)
    except Exception:
        return None


def _index_particle(tensor: Any, particle_idx: Optional[int]) -> Any:
    """Reduce a cost tensor to a single number for reporting.

    Cost values have shape ``(num_particles, ...)``. When ``particle_idx`` is given we take that
    particle (and the max violation over any trailing time dimension, which is how the constraint
    checker aggregates); otherwise we summarize the best (minimum-violation) particle.
    """
    try:
        import torch  # local import: this module stays importable without torch

        if not isinstance(tensor, torch.Tensor):
            return _scalar(tensor)
        t = tensor.detach()
        if t.ndim >= 2 and particle_idx is not None:
            row = t[particle_idx]
            return _scalar(row.max())
        if t.ndim >= 2:
            # Max over trailing dims per particle, then take the best particle.
            flat = t.reshape(t.shape[0], -1).max(dim=1).values
            return _scalar(flat.min())
        if t.ndim == 1 and particle_idx is not None:
            return _scalar(t[particle_idx])
        if t.ndim == 1:
            return _scalar(t.min())
        return _scalar(t)
    except Exception:
        return _scalar(tensor)


def _build_constraint_report(
    cost_dict: Dict[str, dict],
    constraint_checker: Optional[Any],
    particle_idx: Optional[int],
) -> Dict[str, Dict[str, dict]]:
    """Summarize each ``(cost_key, value_name)`` in the cost dict into a compact numeric report."""
    report: Dict[str, Dict[str, dict]] = {}
    for cost_key, cost_info in cost_dict.items():
        values = cost_info.get("values", {})
        entry: Dict[str, dict] = {}
        for name, tensor in values.items():
            value = _index_particle(tensor, particle_idx)
            tol = None
            satisfied = None
            if constraint_checker is not None and cost_info.get("type") == "constraint":
                try:
                    tol = float(constraint_checker._get_tol(cost_key, name))
                    if value is not None:
                        satisfied = bool(value <= tol)
                except Exception:
                    pass
            entry[name] = {"value": value, "tol": tol, "satisfied": satisfied}
        report[cost_key] = {"kind": cost_info.get("type", "unknown"), "values": entry}
    return report


def build_plan_graph(
    plan_skeleton: Sequence[Any],
    *,
    name: str = "tamp_plan",
    env: Optional[Any] = None,
    initial_state: Optional[Any] = None,
    goal_state: Optional[Any] = None,
    cost_dict: Optional[Dict[str, dict]] = None,
    constraint_checker: Optional[Any] = None,
    particle_idx: Optional[int] = None,
    solved: Optional[bool] = None,
    extra_metadata: Optional[dict] = None,
) -> dict:
    """Build a JSON-serializable factor-graph representation of a TAMP plan skeleton.

    Args:
        plan_skeleton: sequence of ``GroundTAMPOperator`` (from the task planner / cuTAMP).
        name: graph name, stored in metadata.
        env: optional ``TAMPEnvironment``; used to annotate object variable nodes with geometry
            (pose/dims) and to record the goal.
        initial_state / goal_state: optional frozensets of ``Atom`` recorded as metadata.
        cost_dict: optional output of ``CostFunction`` for the plan; enables numeric annotation
            of the constraint/cost factors (a per-type report keyed by cost bucket).
        constraint_checker: optional ``ConstraintChecker`` used to attach tolerances and a
            satisfied/violated flag to the numeric report.
        particle_idx: if given, the report reflects this particle; otherwise the best particle.
        solved: optional flag recorded in metadata.
        extra_metadata: optional dict merged into the graph metadata.

    Returns:
        A node-link dict: ``{"format", "name", "directed", "metadata", "nodes", "edges"}``.
    """
    value_to_type = _param_type_map(plan_skeleton)

    # Optional per-object geometry from the environment.
    obj_geometry: Dict[str, dict] = {}
    if env is not None:
        for obj in list(getattr(env, "movables", [])) + list(getattr(env, "statics", [])):
            geo = {}
            for attr in ("pose", "dims", "radius", "height", "color"):
                if getattr(obj, attr, None) is not None:
                    geo[attr] = getattr(obj, attr)
            obj_geometry[obj.name] = geo

    nodes: List[dict] = []
    edges: List[dict] = []
    var_seen: Dict[str, dict] = {}

    def _ensure_var(value: str) -> str:
        """Create (once) and return the id of a variable node for a parameter value."""
        node_id = f"var::{value}"
        if node_id not in var_seen:
            vtype = value_to_type.get(value)
            node = {
                "id": node_id,
                "kind": "variable",
                "name": value,
                "var_type": vtype,
                "var_class": _var_class(vtype),
            }
            if value == "q0":
                node["fixed"] = True  # initial configuration is not optimized
            if value in obj_geometry:
                node["geometry"] = obj_geometry[value]
            var_seen[node_id] = node
            nodes.append(node)
        return node_id

    counts = {"operator": 0, "variable": 0, "constraint": 0, "cost": 0}
    prev_op_id: Optional[str] = None

    for step, ground_op in enumerate(plan_skeleton):
        op_id = f"op::{step}::{ground_op.operator.name}"
        nodes.append(
            {
                "id": op_id,
                "kind": "operator",
                "name": ground_op.operator.name,
                "ground_name": ground_op.name,
                "step": step,
                "values": list(ground_op.values),
                "param_names": [p.name for p in ground_op.operator.parameters],
            }
        )
        counts["operator"] += 1

        # Temporal edge between successive operators.
        if prev_op_id is not None:
            edges.append({"source": prev_op_id, "target": op_id, "rel": "precedes"})
        prev_op_id = op_id

        # Operator -> variable it is parameterized by.
        for value in ground_op.values:
            var_id = _ensure_var(value)
            edges.append({"source": op_id, "target": var_id, "rel": "uses"})

        # Constraint factors owned by this operator.
        for c_idx, con in enumerate(ground_op.constraints):
            ctype = con.type
            con_id = f"con::{step}::{c_idx}::{ctype}"
            con_node = {
                "id": con_id,
                "kind": "constraint",
                "ctype": ctype,
                "params": [str(p) for p in con.params],
                "op_step": step,
                "operator": ground_op.operator.name,
                "removable": CONSTRAINT_REMOVABILITY.get(ctype, {}).get("tier"),
            }
            nodes.append(con_node)
            counts["constraint"] += 1
            edges.append({"source": op_id, "target": con_id, "rel": "owns_constraint"})
            for p in con.params:
                var_id = _ensure_var(str(p))
                edges.append({"source": con_id, "target": var_id, "rel": "constrains"})

        # Cost factors owned by this operator.
        for k_idx, cost in enumerate(ground_op.costs):
            ktype = cost.type
            cost_id = f"cost::{step}::{k_idx}::{ktype}"
            cost_node = {
                "id": cost_id,
                "kind": "cost",
                "ctype": ktype,
                "params": [str(p) for p in cost.params],
                "op_step": step,
                "operator": ground_op.operator.name,
                "removable": CONSTRAINT_REMOVABILITY.get(ktype, {}).get("tier"),
            }
            nodes.append(cost_node)
            counts["cost"] += 1
            edges.append({"source": op_id, "target": cost_id, "rel": "owns_cost"})
            for p in cost.params:
                var_id = _ensure_var(str(p))
                edges.append({"source": cost_id, "target": var_id, "rel": "affects"})

    counts["variable"] = len(var_seen)

    # Optional numeric enrichment: attach a per-type report and stamp factor nodes with the
    # (type-level) numeric summary for their cost bucket.
    report: Optional[dict] = None
    if cost_dict is not None:
        report = _build_constraint_report(cost_dict, constraint_checker, particle_idx)
        for node in nodes:
            if node["kind"] not in ("constraint", "cost"):
                continue
            cost_key = _CTYPE_TO_COST_KEY.get(node["ctype"])
            if cost_key and cost_key in report:
                node["report"] = report[cost_key]
                node["report_scope"] = "type-level"

    metadata: Dict[str, Any] = {
        "num_operators": counts["operator"],
        "operator_sequence": [op.operator.name for op in plan_skeleton],
        "plan_skeleton": [str(op) for op in plan_skeleton],
        "counts": counts,
        "variable_types": {v["name"]: v["var_type"] for v in var_seen.values()},
        "edge_relations": sorted({e["rel"] for e in edges}),
    }
    if solved is not None:
        metadata["solved"] = bool(solved)
    if particle_idx is not None:
        metadata["particle_idx"] = int(particle_idx)
    if initial_state is not None:
        metadata["initial_state"] = sorted(str(a) for a in initial_state)
    if goal_state is not None:
        metadata["goal_state"] = sorted(str(a) for a in goal_state)
    if report is not None:
        metadata["constraint_report"] = report
    if extra_metadata:
        metadata.update(extra_metadata)

    return {
        "format": FORMAT_VERSION,
        "name": name,
        "directed": True,
        "metadata": metadata,
        "nodes": nodes,
        "edges": edges,
    }


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

_KIND_STYLE = {
    "operator": {"shape": "box", "style": "filled", "fillcolor": "#cfe8ff"},
    "constraint": {"shape": "diamond", "style": "filled", "fillcolor": "#ffd7b5"},
    "cost": {"shape": "parallelogram", "style": "filled", "fillcolor": "#e8d7ff"},
    "variable::decision": {"shape": "ellipse", "style": "filled", "fillcolor": "#d7ffd9"},
    "variable::object": {"shape": "doubleoctagon", "style": "filled", "fillcolor": "#fff3b0"},
    "variable::unknown": {"shape": "ellipse", "style": "filled", "fillcolor": "#eeeeee"},
}

# Edge styling. The key to a readable layout is *rank control*: only the operator timeline and
# the operator->factor "owns" edges are allowed to drive Graphviz's ranking. The many-to-many
# variable-sharing edges (uses/constrains/affects) are marked constraint=false so they decorate
# the graph without warping it into a hairball.
_REL_STYLE = {
    "precedes": 'color="#202124" penwidth=2.6 weight=100 arrowsize=0.9',
    "owns_constraint": 'color="#e8710a" penwidth=1.1 weight=8 arrowsize=0.5',
    "owns_cost": 'color="#7b3ff2" penwidth=1.1 weight=8 arrowsize=0.5',
    "uses": 'color="#9aa0a6" style=dashed arrowhead=none constraint=true weight=1',
    "constrains": 'color="#f4b400" style=dotted arrowhead=none constraint=false penwidth=0.8',
    "affects": 'color="#b39ddb" style=dotted arrowhead=none constraint=false penwidth=0.8',
}


def _node_label(node: dict, compact: bool = False) -> str:
    if node["kind"] == "operator":
        return node["ground_name"]
    if node["kind"] in ("constraint", "cost"):
        # In compact (DOT) mode drop the params — they are shown as edges to the variable nodes.
        label = node["ctype"] if compact else f"{node['ctype']}({', '.join(node['params'])})"
        report = node.get("report")
        if report:
            flags = []
            for vname, info in report.get("values", {}).items():
                sat = info.get("satisfied")
                val = info.get("value")
                if val is not None:
                    mark = "" if sat is None else ("✓" if sat else "✗")
                    flags.append(f"{vname}={val:.3g}{mark}")
            if flags:
                label += "\\n" + ", ".join(flags)
        return label
    return node["name"]  # variable


def _dot_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace('"', '\\"')


def _dot_node_line(node: dict) -> str:
    if node["kind"] == "variable":
        style_key = f"variable::{node.get('var_class', 'unknown')}"
    else:
        style_key = node["kind"]
    style = dict(_KIND_STYLE.get(style_key, _KIND_STYLE["variable::unknown"]))
    if node["kind"] in ("constraint", "cost"):
        style["fontsize"] = "9"
    label = _dot_escape(_node_label(node, compact=True)).replace("\\\\n", "\\n")
    attr = " ".join(f'{k}="{v}"' for k, v in style.items())
    return f'  "{node["id"]}" [label="{label}" {attr}];'


def plan_graph_to_dot(graph: dict, cluster: bool = True) -> str:
    """Render the plan graph as Graphviz DOT source (renderable with ``dot -Tpng``).

    Args:
        cluster: if True, box each operator together with the constraints and costs it owns, so
            the plan reads as a left-to-right sequence of grouped steps.
    """
    lines = [f'digraph "{_dot_escape(graph.get("name", "tamp_plan"))}" {{']
    lines.append("  rankdir=LR;")
    lines.append("  splines=spline;")
    lines.append("  nodesep=0.28;")
    lines.append("  ranksep=1.0;")
    lines.append("  newrank=true;")
    lines.append('  node [fontname="Helvetica" fontsize=10 margin="0.06,0.04"];')
    lines.append('  edge [fontname="Helvetica" fontsize=8];')

    nodes_by_id = {n["id"]: n for n in graph["nodes"]}
    variables = [n for n in graph["nodes"] if n["kind"] == "variable"]

    if cluster:
        # Variable nodes must be declared at top level so they are not captured by a cluster.
        for node in variables:
            lines.append(_dot_node_line(node))

        # One cluster per plan step: the operator plus the factors it owns.
        by_step: Dict[int, Dict[str, Any]] = {}
        for node in graph["nodes"]:
            if node["kind"] == "operator":
                by_step.setdefault(node["step"], {"op": None, "factors": []})["op"] = node
            elif node["kind"] in ("constraint", "cost"):
                by_step.setdefault(node["op_step"], {"op": None, "factors": []})["factors"].append(node)

        for step in sorted(by_step):
            entry = by_step[step]
            op = entry["op"]
            op_name = op["name"] if op else ""
            lines.append(f"  subgraph cluster_step_{step} {{")
            lines.append(f'    label="{step} · {op_name}"; fontsize=10; fontname="Helvetica";')
            lines.append('    style="rounded,filled"; color="#d0d0d0"; fillcolor="#fafafa"; margin=8;')
            if op:
                lines.append("  " + _dot_node_line(op))
            for f in entry["factors"]:
                lines.append("  " + _dot_node_line(f))
            lines.append("  }")
    else:
        for node in graph["nodes"]:
            lines.append(_dot_node_line(node))

    for edge in graph["edges"]:
        style = _REL_STYLE.get(edge["rel"], "")
        lines.append(f'  "{edge["source"]}" -> "{edge["target"]}" [{style}];')

    lines.append("}")
    return "\n".join(lines)


_HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>__TITLE__</title>
__VIS_SCRIPT__
<style>
  :root{
    --panel: 400px; --bg:#f5f6f8; --card:#ffffff; --line:#e7e9ee; --ink:#1f2430; --muted:#6b7280;
    --op:#2b7de9; --con:#e8710a; --cost:#8b5cf6; --dec:#12924f; --obj:#e0a400;
    --font: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
    --mono: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  }
  *{box-sizing:border-box}
  html,body{margin:0;height:100%;font-family:var(--font);color:var(--ink);background:var(--bg);
            -webkit-font-smoothing:antialiased}
  #app{display:flex;height:100vh}
  #stage{flex:1;display:flex;flex-direction:column;min-width:0}
  #topbar{display:flex;align-items:center;justify-content:space-between;gap:12px;padding:10px 14px;
          background:var(--card);border-bottom:1px solid var(--line)}
  .brand{display:flex;align-items:center;gap:10px;min-width:0}
  .brand .glyph{width:26px;height:26px;border-radius:8px;flex:none;
     background:conic-gradient(from 210deg,var(--op),var(--dec),var(--obj),var(--con),var(--cost),var(--op))}
  .brand .t{font-weight:650;font-size:15px;line-height:1.1;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
  .brand .s{font-size:12px;color:var(--muted)}
  .tools{display:flex;align-items:center;gap:8px;flex:none}
  .seg{display:inline-flex;border:1px solid var(--line);border-radius:9px;overflow:hidden;background:#fff}
  .seg button{border:0;background:#fff;padding:7px 12px;font:inherit;font-size:12px;cursor:pointer;color:var(--muted)}
  .seg button.on{background:var(--op);color:#fff;font-weight:600}
  .btn{border:1px solid var(--line);background:#fff;border-radius:9px;padding:7px 12px;font:inherit;font-size:12px;
       cursor:pointer;color:var(--ink)}
  .btn:hover{background:#f1f3f7}
  .btn.on{background:var(--op);color:#fff;border-color:var(--op)}
  #stage{position:relative}
  #remlegend{position:absolute;left:14px;bottom:44px;background:rgba(255,255,255,0.96);border:1px solid var(--line);
     border-radius:10px;padding:9px 12px;box-shadow:0 2px 10px rgba(20,25,40,.08);display:none;font-size:12px;z-index:5}
  #remlegend .rl-title{font-weight:650;margin-bottom:6px}
  #remlegend .rl-row{display:flex;align-items:center;gap:8px;margin:4px 0}
  #remlegend .rl-sw{width:16px;height:16px;border-radius:4px;flex:none}
  .remchip{display:inline-block;padding:2px 9px;border-radius:999px;color:#fff;font-size:11px;font-weight:600}
  #search{border:1px solid var(--line);border-radius:9px;padding:7px 10px;font:inherit;font-size:12px;width:150px;
          background:#fff}
  #search:focus{outline:2px solid #cfe0fb;border-color:var(--op)}
  #network{flex:1;min-height:0;background:
     radial-gradient(circle at 1px 1px,#e9ebf0 1px,transparent 0) 0 0/22px 22px, #fbfcfe}
  #hintbar{padding:6px 14px;font-size:11px;color:var(--muted);background:var(--card);border-top:1px solid var(--line)}

  #panel{width:var(--panel);flex:none;height:100%;display:flex;flex-direction:column;background:var(--card);
         border-left:1px solid var(--line)}
  .tabs{display:flex;gap:2px;padding:10px 12px 0;border-bottom:1px solid var(--line);flex:none}
  .tabs button{border:0;background:transparent;font:inherit;font-size:13px;color:var(--muted);cursor:pointer;
               padding:8px 12px;border-radius:8px 8px 0 0;border-bottom:2px solid transparent}
  .tabs button.on{color:var(--ink);font-weight:600;border-bottom-color:var(--op)}
  .tabwrap{flex:1;overflow-y:auto;padding:14px}
  .tabpane{display:none}
  .tabpane.on{display:block}
  h3{font-size:11px;text-transform:uppercase;letter-spacing:.05em;color:var(--muted);margin:16px 0 8px}
  h3:first-child{margin-top:0}
  .muted{color:var(--muted);font-size:12px}

  .empty{color:var(--muted);font-size:13px;padding:20px 4px;text-align:center}
  .dtitle{display:flex;align-items:center;gap:8px;margin-bottom:10px;flex-wrap:wrap}
  .dtitle .name{font-weight:650;font-size:15px;word-break:break-word}
  .chip{display:inline-block;padding:2px 9px;border-radius:999px;color:#fff;font-size:11px;font-weight:600;
        text-transform:capitalize}
  .kv{width:100%;border-collapse:collapse}
  .kv td{padding:5px 6px;vertical-align:top;border-top:1px solid #f1f2f5}
  .kv tr:first-child td{border-top:0}
  .kv td.k{color:var(--muted);width:34%;font-size:12px}
  .kv td.v{font-family:var(--mono);font-size:12px;word-break:break-word}
  .val{font-family:var(--mono);color:var(--dec)}
  .ok{color:#137333}.bad{color:#c5221f}
  .card{background:#fbfcfe;border:1px solid var(--line);border-radius:10px;padding:12px;margin-top:12px}
  .card .lead{font-size:13.5px;font-weight:600;margin-bottom:3px}
  .card .prose{font-size:12.5px;color:#3c4043;line-height:1.5}
  .ptable{width:100%;border-collapse:collapse;margin-top:8px}
  .ptable td{padding:6px 6px;border-top:1px solid #eef0f3;vertical-align:top}
  .ptable td.p{width:38%}
  .ptable .pname{font-weight:600;font-size:12.5px}
  .ptable .pval{font-family:var(--mono);font-size:11.5px;color:var(--dec);display:block;margin-top:1px}
  .ptable td.d{font-size:12.5px;color:#3c4043;line-height:1.45}
  pre{background:#f6f8fa;border-radius:8px;padding:8px;font-size:11px;overflow-x:auto;margin:6px 0 0}

  .legrow{display:flex;align-items:center;gap:9px;font-size:12.5px;padding:6px 8px;border-radius:8px;cursor:pointer;
          user-select:none}
  .legrow:hover{background:#f3f5f9}
  .legrow.off{opacity:.42}
  .legrow.off .name{text-decoration:line-through}
  .swatch{width:20px;height:14px;border-radius:4px;border:1.5px solid #999;flex:none}
  .eline{width:26px;flex:none;border-top-width:3px;border-top-style:solid}
  .legrow .name{flex:1}
  .legrow .cnt{color:var(--muted);font-size:11px;font-variant-numeric:tabular-nums}

  details.ref{border:1px solid var(--line);border-radius:10px;padding:8px 11px;margin-bottom:7px;background:#fff}
  details.ref>summary{cursor:pointer;font-size:12.5px;list-style:none}
  details.ref>summary::-webkit-details-marker{display:none}
  details.ref>summary::before{content:"▸";color:var(--muted);margin-right:7px;display:inline-block;transition:.15s}
  details.ref[open]>summary::before{transform:rotate(90deg)}
  details.ref .rlead{font-weight:600}
  details.ref .rprose{font-size:12px;color:#3c4043;margin:5px 0;line-height:1.45}
  .refkind{display:inline-block;width:10px;height:10px;border-radius:3px;margin-right:7px;vertical-align:-1px}
</style>
</head>
<body>
<div id="app">
  <section id="stage">
    <header id="topbar">
      <div class="brand">
        <span class="glyph"></span>
        <div>
          <div class="t">__TITLE__</div>
          <div class="s" id="summary"></div>
        </div>
      </div>
      <div class="tools">
        <div class="seg" id="layoutSeg">
          <button data-mode="timeline" class="on">Timeline</button>
          <button data-mode="force">Force</button>
        </div>
        <input id="search" placeholder="Search node…" autocomplete="off" spellcheck="false" />
        <button id="btn-removable" class="btn" title="Highlight which constraints can be removed to change the data distribution">Removable</button>
        <button id="btn-fit" class="btn">Fit</button>
        <button id="btn-physics" class="btn" hidden>Freeze</button>
        <button id="btn-reset" class="btn">Clear</button>
      </div>
    </header>
    <div id="network"></div>
    <div id="remlegend">
      <div class="rl-title">Constraint removability</div>
      <div class="rl-row"><span class="rl-sw" style="background:#0f9d58"></span>removable — changes the data distribution</div>
      <div class="rl-row"><span class="rl-sw" style="background:#f4b400"></span>no-op — relax via tolerance instead</div>
      <div class="rl-row"><span class="rl-sw" style="background:#db4437"></span>structural — required, can't remove</div>
      <div class="rl-title" style="font-weight:400;color:#6b7280;margin-top:6px">Other nodes dimmed. Click a factor for details.</div>
    </div>
    <div id="hintbar">Click a node or edge for details · drag to rearrange · scroll to zoom · use Legend&nbsp;&amp;&nbsp;filters to declutter</div>
  </section>

  <aside id="panel">
    <nav class="tabs">
      <button data-tab="details" class="on">Details</button>
      <button data-tab="legend">Legend &amp; filters</button>
      <button data-tab="reference">Reference</button>
    </nav>
    <div class="tabwrap">
      <div class="tabpane on" id="tab-details">
        <div id="details"><div class="empty">Click a node or edge to inspect it.<br>Selecting a node highlights its neighborhood.</div></div>
      </div>
      <div class="tabpane" id="tab-legend">
        <div class="muted">Click a row to show / hide that kind in the graph.</div>
        <h3>Nodes</h3><div id="legend-nodes"></div>
        <h3>Edges</h3><div id="legend-edges"></div>
        <h3>Plan metadata</h3><div id="meta"></div>
      </div>
      <div class="tabpane" id="tab-reference">
        <div class="muted">Documentation for every operator, constraint and cost in the TAMP domain. Click to expand.</div>
        <div id="reference" style="margin-top:10px"></div>
      </div>
    </div>
  </aside>
</div>

<script>
const GRAPH = __GRAPH_JSON__;
const DOCS = __DOCS_JSON__;

// ---- style maps (kept in sync with the Python DOT styling) -------------------
const NODE_STYLE = {
  operator:  {shape:"box",    color:"#dbe9ff", border:"#2b7de9", label:"operator (task step)"},
  constraint:{shape:"diamond",color:"#ffe0c2", border:"#e8710a", label:"constraint (hard factor)"},
  cost:      {shape:"hexagon",color:"#ece1ff", border:"#8b5cf6", label:"cost (soft factor)"},
  "variable:decision":{shape:"dot", color:"#c9f2d8", border:"#12924f", label:"decision var (conf / pose / grasp / traj)"},
  "variable:object":  {shape:"star",color:"#ffeeb0", border:"#e0a400", label:"object var (toy / plate / surface)"},
  "variable:unknown": {shape:"dot", color:"#e8eaed", border:"#9aa0a6", label:"variable"},
};
const EDGE_STYLE = {
  precedes:        {color:"#3c4043", width:2.6, dashes:false,  arrows:true,  physics:true,  op:0.95, label:"precedes (plan order)"},
  owns_constraint: {color:"#e8710a", width:1.4, dashes:false,  arrows:true,  physics:true,  op:0.85, label:"operator owns constraint"},
  owns_cost:       {color:"#8b5cf6", width:1.4, dashes:false,  arrows:true,  physics:true,  op:0.85, label:"operator owns cost"},
  uses:            {color:"#9aa0a6", width:1,   dashes:[6,6],  arrows:false, physics:false, op:0.5,  label:"operator uses variable"},
  constrains:      {color:"#e0a400", width:1,   dashes:[2,4],  arrows:false, physics:false, op:0.45, label:"constraint scopes variable"},
  affects:         {color:"#b39ddb", width:1,   dashes:[2,4],  arrows:false, physics:false, op:0.45, label:"cost scopes variable"},
};

// Removability highlight colors (see DOCS.removability for the per-type effect text).
const REMOVABLE_STYLE = {
  free:      {border:"#0f9d58", bg:"#d6f5e3", label:"removable"},
  noop:      {border:"#f4b400", bg:"#fff3cd", label:"no-op to remove"},
  structural:{border:"#db4437", bg:"#fbdcd9", label:"structural — keep"},
};

function nodeStyleKey(n){ return n.kind==="variable" ? "variable:"+(n.var_class||"unknown") : n.kind; }
function nodeLabel(n){
  if(n.kind==="operator") return n.ground_name;
  if(n.kind==="constraint"||n.kind==="cost") return n.ctype;
  return n.name;
}
const nodeById = {};
GRAPH.nodes.forEach(n=>nodeById[n.id]=n);

// ---- hierarchical levels (timeline layout) -----------------------------------
const varFirstStep = {};
GRAPH.edges.forEach(e=>{
  const src = nodeById[e.source];
  if(src && src.kind==="operator" && (e.rel==="uses")){
    varFirstStep[e.target] = Math.min(varFirstStep[e.target] ?? Infinity, src.step);
  }
});
function levelOf(n){
  if(n.kind==="operator") return 2*n.step;
  if(n.kind==="constraint"||n.kind==="cost") return 2*n.op_step;
  const s = varFirstStep[n.id]; return (s===undefined?0:2*s)+1;
}

// ---- build datasets ----------------------------------------------------------
function nodeTitle(n){
  if(n.kind==="operator") return n.ground_name;
  if(n.kind==="constraint"||n.kind==="cost") return `${n.ctype}(${(n.params||[]).join(", ")})`;
  return `${n.name} : ${n.var_type||"variable"}`;
}
const visNodes = GRAPH.nodes.map(n=>{
  const s = NODE_STYLE[nodeStyleKey(n)] || NODE_STYLE["variable:unknown"];
  const isOp = n.kind==="operator";
  // Short on-graph label for operators ("step · Op") to keep boxes compact; the full grounding is
  // available on hover (title) and in the Details panel. Keeps the timeline layout from overlapping.
  const label = isOp ? (n.step + " · " + n.name) : nodeLabel(n);
  return {
    id:n.id, label:label, title:nodeTitle(n), shape:s.shape, level:levelOf(n),
    color:{background:s.color, border:s.border, highlight:{background:s.color, border:"#111"}},
    size: isOp?18:(n.kind==="variable"?11:12),
    font:{size:isOp?13:11.5, color:"#1f2430", strokeWidth: isOp?0:3, strokeColor:"#ffffff"},
    margin: isOp?7:5, opacity:1,
  };
});
const edgeRecords = GRAPH.edges.map((e,i)=>{
  const s = EDGE_STYLE[e.rel] || {color:"#999",width:1,dashes:false,arrows:false,physics:true,op:0.6};
  return {
    id:"e"+i, from:e.source, to:e.target,
    color:{color:s.color, highlight:"#111", opacity:s.op},
    width:s.width, dashes:s.dashes, arrows: s.arrows? {to:{enabled:true,scaleFactor:0.6}} : undefined,
    physics:s.physics, smooth:{enabled:true,type:"cubicBezier",roundness:0.4},
    _base:{color:s.color, op:s.op, width:s.width}, _rel:e.rel,
  };
});
const nodes = new vis.DataSet(visNodes);
const edges = new vis.DataSet(edgeRecords);
const container = document.getElementById("network");

let layoutMode = "timeline";
let physicsOn = false;
let network = null;
let removableOn = false;
// Filter state (declared before buildNetwork so applyFilters can read it during initial build).
const hiddenNodeKinds = new Set(), hiddenRels = new Set();

function buildOptions(mode){
  const base = {
    interaction:{hover:true, tooltipDelay:150, multiselect:false, hideEdgesOnDrag:false, navigationButtons:false, keyboard:false},
    nodes:{borderWidth:1.6, borderWidthSelected:3, shadow:{enabled:true,size:5,x:0,y:1,color:"rgba(20,25,40,0.10)"},
           shapeProperties:{borderRadius:7}},
    edges:{selectionWidth:1.6, arrowStrikethrough:false},
  };
  if(mode==="timeline"){
    base.layout={hierarchical:{enabled:true, direction:"LR", sortMethod:"directed",
                  levelSeparation:200, nodeSpacing:120, treeSpacing:210, blockShifting:true,
                  edgeMinimization:true, parentCentralization:false, shakeTowards:"roots"}};
    // Hierarchical repulsion keeps same-level nodes from overlapping without moving them off their
    // levels; a short stabilization settles it then we freeze.
    base.physics={enabled:true, solver:"hierarchicalRepulsion",
      hierarchicalRepulsion:{nodeDistance:130, springLength:120, avoidOverlap:1},
      stabilization:{iterations:180, fit:true}};
  } else {
    base.layout={hierarchical:{enabled:false}};
    base.physics={enabled:true, solver:"forceAtlas2Based",
      forceAtlas2Based:{gravitationalConstant:-72, centralGravity:0.012, springLength:95, springConstant:0.08, avoidOverlap:0.7},
      stabilization:{iterations:320}};
  }
  return base;
}

function wireNetwork(){
  network.on("selectNode", p=>{ const n=nodeById[p.nodes[0]]; showNode(n); if(!removableOn) setHighlight(n.id); switchTab("details"); });
  network.on("selectEdge", p=>{ if(p.nodes.length) return; const r=edges.get(p.edges[0]); if(r){ showEdge(GRAPH.edges[+r.id.slice(1)]); switchTab("details"); } });
  network.on("deselectNode", ()=>{ if(!removableOn) setHighlight(null); });
  network.on("click", p=>{ if(!p.nodes.length && !p.edges.length){ if(!removableOn){ setHighlight(null); clearDetails(); } } });
  // Both layouts run a short physics pass to resolve overlaps, then freeze so the graph stays put.
  physicsOn = (layoutMode==="force");
  syncPhysicsBtn();
  network.once("stabilizationIterationsDone", ()=>{
    network.setOptions({physics:false});
    if(layoutMode==="force"){ physicsOn=false; syncPhysicsBtn(); }
  });
}
function buildNetwork(){
  network = new vis.Network(container, {nodes, edges}, buildOptions(layoutMode));
  wireNetwork();
  applyFilters();
}
buildNetwork();

// ---- selection highlighting --------------------------------------------------
function setHighlight(nodeId){
  if(!nodeId){
    nodes.update(GRAPH.nodes.map(n=>({id:n.id, opacity:1})));
    edges.update(edgeRecords.map(r=>({id:r.id, color:{color:r._base.color, highlight:"#111", opacity:r._base.op}, width:r._base.width})));
    return;
  }
  const keep=new Set([nodeId]); const keepE=new Set();
  GRAPH.edges.forEach((e,i)=>{ if(e.source===nodeId||e.target===nodeId){ keep.add(e.source); keep.add(e.target); keepE.add("e"+i); } });
  nodes.update(GRAPH.nodes.map(n=>({id:n.id, opacity: keep.has(n.id)?1:0.12})));
  edges.update(edgeRecords.map(r=>({id:r.id,
    color:{color: keepE.has(r.id)?"#111":r._base.color, highlight:"#111", opacity: keepE.has(r.id)?0.95:0.05},
    width: keepE.has(r.id)? r._base.width+1 : r._base.width})));
}

// ---- filters -----------------------------------------------------------------
function applyFilters(){
  nodes.update(GRAPH.nodes.map(n=>({id:n.id, hidden: hiddenNodeKinds.has(nodeStyleKey(n))})));
  edges.update(edgeRecords.map(r=>({id:r.id, hidden: hiddenRels.has(r._rel)})));
}

// ---- removability highlight --------------------------------------------------
// Colors each constraint/cost by whether it can be removed (green=free, amber=no-op, red=structural)
// and dims everything else, so you can see at a glance which constraints move the data distribution.
function setRemovableMode(on){
  removableOn = on;
  document.getElementById("btn-removable").classList.toggle("on", on);
  document.getElementById("remlegend").style.display = on ? "block" : "none";
  if(!on){
    nodes.update(visNodes.map(v=>({id:v.id, color:v.color, borderWidth:1.6, opacity:1, shapeProperties:{borderRadius:7, borderDashes:false}})));
    return;
  }
  nodes.update(GRAPH.nodes.map(n=>{
    if(n.kind==="constraint"||n.kind==="cost"){
      const st = REMOVABLE_STYLE[n.removable||"structural"] || REMOVABLE_STYLE.structural;
      return {id:n.id, opacity:1, borderWidth:4,
        color:{background:st.bg, border:st.border, highlight:{background:st.bg, border:"#111"}},
        shapeProperties:{borderRadius:7, borderDashes: (n.removable==="noop")?[4,3]:false}};
    }
    return {id:n.id, opacity:0.15};
  }));
}

// ---- details panel -----------------------------------------------------------
function esc(v){ return String(v).replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;"); }
function row(k,v){ return `<tr><td class="k">${esc(k)}</td><td class="v">${v}</td></tr>`; }
function clearDetails(){ document.getElementById("details").innerHTML='<div class="empty">Click a node or edge to inspect it.<br>Selecting a node highlights its neighborhood.</div>'; }

function reportHtml(report){
  let rows=""; const vals=(report&&report.values)||{};
  for(const [name,info] of Object.entries(vals)){
    let cls="",mark=""; if(info.satisfied===true){cls="ok";mark=" ✓";} else if(info.satisfied===false){cls="bad";mark=" ✗";}
    const val=(info.value==null)?"–":(+info.value).toPrecision(4);
    const tol=(info.tol==null)?"":` (tol ${info.tol})`;
    rows+=`<tr><td class="k">${esc(name)}</td><td class="v ${cls}">${val}${tol}${mark}</td></tr>`;
  }
  return rows?`<table class="kv">${rows}</table>`:"";
}

function docBlock(n){
  let doc, spec, vals;
  if(n.kind==="operator"){ doc=DOCS.operators[n.name]; if(!doc) return "";
    spec=(n.param_names||[]).map(nm=>[nm,(doc.params||{})[nm]||""]); vals=n.values||[]; }
  else if(n.kind==="constraint"||n.kind==="cost"){ doc=(n.kind==="constraint"?DOCS.constraints:DOCS.costs)[n.ctype]; if(!doc) return "";
    spec=doc.params||[]; vals=n.params||[]; }
  else return "";
  let h=`<div class="card"><div class="lead">${esc(doc.summary||"")}</div>`;
  if(doc.does||doc.detail) h+=`<div class="prose">${esc(doc.does||doc.detail)}</div>`;
  if(spec&&spec.length){
    let pr="";
    for(let i=0;i<spec.length;i++){
      const val=vals[i]!==undefined?vals[i]:"";
      pr+=`<tr><td class="p"><span class="pname">${esc(spec[i][0])}</span><span class="pval">${esc(val)}</span></td><td class="d">${esc(spec[i][1])}</td></tr>`;
    }
    h+=`<table class="ptable">${pr}</table>`;
  }
  return h+`</div>`;
}

// Removability card: how removing this constraint/cost affects the sampled data distribution.
function removeBlock(n){
  if(n.kind!=="constraint" && n.kind!=="cost") return "";
  const rem=(DOCS.removability||{})[n.ctype]; if(!rem) return "";
  const st=REMOVABLE_STYLE[rem.tier]||REMOVABLE_STYLE.structural;
  return `<div class="card" style="border-color:${st.border}">`
    +`<div class="lead">Removability &nbsp;<span class="remchip" style="background:${st.border}">${esc(rem.label)}</span></div>`
    +`<div class="prose" style="margin-top:6px">${esc(rem.effect)}</div></div>`;
}

function showNode(n){
  const s=NODE_STYLE[nodeStyleKey(n)]||NODE_STYLE["variable:unknown"];
  let html=`<div class="dtitle"><span class="chip" style="background:${s.border}">${esc(n.kind)}</span><span class="name">${esc(nodeLabel(n))}</span></div>`;
  let rows="";
  if(n.kind==="operator"){
    rows+=row("step",n.step); rows+=row("operator",esc(n.name));
    rows+=row("grounding",esc(n.ground_name)); rows+=row("parameters",(n.values||[]).map(esc).join(", "));
  } else if(n.kind==="constraint"||n.kind==="cost"){
    rows+=row("type",esc(n.ctype)); rows+=row("owner op",`step ${n.op_step} · ${esc(n.operator)}`);
    rows+=row("scope",(n.params||[]).map(esc).join(", "));
    rows+=row("role", n.kind==="constraint"?"hard — must be satisfied":"soft — minimized");
  } else {
    rows+=row("variable",esc(n.name)); rows+=row("type",esc(n.var_type)); rows+=row("class",esc(n.var_class));
    if(n.fixed) rows+=row("fixed","yes (not optimized)");
    if(n.geometry) rows+=row("geometry",`<pre>${esc(JSON.stringify(n.geometry))}</pre>`);
    const touched=GRAPH.edges.filter(e=>e.target===n.id).map(e=>`${e.rel}: ${nodeLabel(nodeById[e.source])}`);
    if(touched.length) rows+=row("referenced by",touched.map(esc).join("<br>"));
  }
  html+=`<table class="kv">${rows}</table>`;
  const rep=n.report?reportHtml(n.report):"";
  if(rep) html+=`<h3>values (${esc(n.report_scope||"")})</h3>`+rep;
  html+=docBlock(n);
  html+=removeBlock(n);
  document.getElementById("details").innerHTML=html;
}

function showEdge(e){
  const s=EDGE_STYLE[e.rel]||{}; const from=nodeById[e.source], to=nodeById[e.target];
  let html=`<div class="dtitle"><span class="chip" style="background:${s.color||'#999'}">edge</span><span class="name">${esc(e.rel)}</span></div>`;
  html+=`<table class="kv">`+row("meaning",esc(s.label||e.rel))
       +row("from",`${esc(from.kind)} · ${esc(nodeLabel(from))}`)
       +row("to",`${esc(to.kind)} · ${esc(nodeLabel(to))}`)+`</table>`;
  document.getElementById("details").innerHTML=html;
}

// ---- tabs --------------------------------------------------------------------
function switchTab(name){
  document.querySelectorAll(".tabs button").forEach(b=>b.classList.toggle("on",b.dataset.tab===name));
  document.querySelectorAll(".tabpane").forEach(p=>p.classList.toggle("on",p.id==="tab-"+name));
}
document.querySelectorAll(".tabs button").forEach(b=>b.onclick=()=>switchTab(b.dataset.tab));

// ---- controls ----------------------------------------------------------------
function syncPhysicsBtn(){ const b=document.getElementById("btn-physics"); b.hidden=(layoutMode!=="force"); b.textContent=physicsOn?"Freeze":"Resume physics"; }
document.getElementById("btn-physics").onclick=()=>{ physicsOn=!physicsOn; network.setOptions({physics:physicsOn}); syncPhysicsBtn(); };
document.getElementById("btn-fit").onclick=()=>network.fit({animation:true});
document.getElementById("btn-reset").onclick=()=>{ network.unselectAll(); if(removableOn) setRemovableMode(false); setHighlight(null); clearDetails(); };
document.getElementById("btn-removable").onclick=()=>{ if(!removableOn){ network.unselectAll(); setHighlight(null); } setRemovableMode(!removableOn); };
document.querySelectorAll("#layoutSeg button").forEach(b=>b.onclick=()=>{
  if(b.dataset.mode===layoutMode) return;
  layoutMode=b.dataset.mode;
  document.querySelectorAll("#layoutSeg button").forEach(x=>x.classList.toggle("on",x===b));
  network.destroy(); buildNetwork(); syncPhysicsBtn();
  if(removableOn) setRemovableMode(true);
  network.once("afterDrawing",()=>network.fit());
});
syncPhysicsBtn();

// search: jump to first node whose label matches
const search=document.getElementById("search");
function doSearch(){
  const q=search.value.trim().toLowerCase(); if(!q) return;
  const hit=GRAPH.nodes.find(n=>nodeLabel(n).toLowerCase().includes(q) || n.id.toLowerCase().includes(q));
  if(hit){ network.selectNodes([hit.id]); showNode(hit); setHighlight(hit.id); switchTab("details");
           network.focus(hit.id,{scale:1.1,animation:true}); }
}
search.addEventListener("keydown",e=>{ if(e.key==="Enter") doSearch(); });

// ---- legend (with show/hide filters) -----------------------------------------
const nodeCounts={}, relCounts={};
GRAPH.nodes.forEach(n=>{const k=nodeStyleKey(n);nodeCounts[k]=(nodeCounts[k]||0)+1;});
GRAPH.edges.forEach(e=>relCounts[e.rel]=(relCounts[e.rel]||0)+1);

const legN=document.getElementById("legend-nodes");
Object.entries(NODE_STYLE).filter(([k])=>nodeCounts[k]).forEach(([k,s])=>{
  const el=document.createElement("div"); el.className="legrow";
  el.innerHTML=`<span class="swatch" style="background:${s.color};border-color:${s.border}"></span><span class="name">${esc(s.label)}</span><span class="cnt">${nodeCounts[k]}</span>`;
  el.onclick=()=>{ if(hiddenNodeKinds.has(k))hiddenNodeKinds.delete(k); else hiddenNodeKinds.add(k); el.classList.toggle("off"); applyFilters(); };
  legN.appendChild(el);
});
const legE=document.getElementById("legend-edges");
Object.entries(EDGE_STYLE).filter(([k])=>relCounts[k]).forEach(([k,s])=>{
  const style=Array.isArray(s.dashes)?"dashed":"solid";
  const el=document.createElement("div"); el.className="legrow";
  el.innerHTML=`<span class="eline" style="border-top-color:${s.color};border-top-style:${style}"></span><span class="name">${esc(s.label)}${s.arrows?" →":""}</span><span class="cnt">${relCounts[k]}</span>`;
  el.onclick=()=>{ if(hiddenRels.has(k))hiddenRels.delete(k); else hiddenRels.add(k); el.classList.toggle("off"); applyFilters(); };
  legE.appendChild(el);
});

// ---- summary + metadata ------------------------------------------------------
const m=GRAPH.metadata||{}, c=m.counts||{};
document.getElementById("summary").innerHTML=
  `${(m.operator_sequence||[]).length} steps · ${c.variable||0} vars · ${c.constraint||0} constraints · ${c.cost||0} costs`
  +(m.solved===undefined?"":(m.solved?' · <span class="ok">solved</span>':' · <span class="bad">unsolved</span>'));
let mr=""; mr+=row("plan",(m.operator_sequence||[]).map(esc).join(" → "));
if(m.goal_state) mr+=row("goal",m.goal_state.map(esc).join("<br>"));
if(m.mode) mr+=row("mode",esc(m.mode));
document.getElementById("meta").innerHTML=`<table class="kv">${mr}</table>`;

// ---- reference / glossary ----------------------------------------------------
function refParamsObj(o){ return Object.entries(o||{}).map(([k,v])=>`<tr><td class="p"><span class="pname">${esc(k)}</span></td><td class="d">${esc(v)}</td></tr>`).join(""); }
function refParamsList(l){ return (l||[]).map(([k,v])=>`<tr><td class="p"><span class="pname">${esc(k)}</span></td><td class="d">${esc(v)}</td></tr>`).join(""); }
function refEntry(name,doc,params,color,ctype){
  const desc=esc(doc.does||doc.detail||"");
  const rem=ctype?(DOCS.removability||{})[ctype]:null;
  let remHtml="";
  if(rem){ const st=REMOVABLE_STYLE[rem.tier]||REMOVABLE_STYLE.structural;
    remHtml=`<div class="rprose" style="margin-top:6px"><span class="remchip" style="background:${st.border}">${esc(rem.label)}</span> ${esc(rem.effect)}</div>`; }
  return `<details class="ref"><summary><span class="refkind" style="background:${color}"></span><span class="rlead">${esc(name)}</span> — ${esc(doc.summary||"")}</summary>`
    +(desc?`<div class="rprose">${desc}</div>`:"")+remHtml
    +(params?`<table class="ptable">${params}</table>`:"")+`</details>`;
}
let ref=`<h3>Operators</h3>`;
for(const [k,v] of Object.entries(DOCS.operators)) ref+=refEntry(k,v,refParamsObj(v.params),"#2b7de9");
ref+=`<h3>Constraints — hard factors</h3>`;
for(const [k,v] of Object.entries(DOCS.constraints)) ref+=refEntry(k,v,refParamsList(v.params),"#e8710a",k);
ref+=`<h3>Costs — soft factors</h3>`;
for(const [k,v] of Object.entries(DOCS.costs)) ref+=refEntry(k,v,refParamsList(v.params),"#8b5cf6",k);
document.getElementById("reference").innerHTML=ref;

network.once("afterDrawing",()=>network.fit());
</script>
</body>
</html>
"""


_VIS_CDN_URL = "https://unpkg.com/vis-network@9.1.9/standalone/umd/vis-network.min.js"


def plan_graph_to_html(graph: dict, inline_vis: Optional[Union[str, Path]] = None) -> str:
    """Render the plan graph as an interactive HTML page (clickable nodes/edges + legend + panel).

    Args:
        inline_vis: to make the page fully self-contained (viewable offline), pass either the
            vis-network UMD JavaScript source as a string, or a path to a ``.js`` file to inline.
            When omitted, the page loads vis-network from a CDN (needs internet to view).
    """
    # Only the literal "</script" can prematurely close the embedding <script> block; neutralize
    # just that (case-insensitively). A blanket "</"->"<\/" replacement would corrupt regex/string
    # literals inside the vis-network library.
    def _script_safe(text: str) -> str:
        return re.sub(r"</(script)", r"<\\/\1", text, flags=re.IGNORECASE)

    graph_json = _script_safe(json.dumps(graph, default=str))
    docs_json = _script_safe(json.dumps(PLAN_DOCS))
    raw_name = str(graph.get("name", "TAMP plan"))
    title = raw_name.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;") + " — plan graph"

    if inline_vis is not None:
        js = inline_vis
        if isinstance(js, Path) or (isinstance(js, str) and len(js) < 4096 and Path(js).exists()):
            js = Path(js).read_text(encoding="utf-8")
        vis_script = "<script>" + _script_safe(js) + "</script>"
    else:
        vis_script = f'<script src="{_VIS_CDN_URL}"></script>'

    return (
        _HTML_TEMPLATE.replace("__VIS_SCRIPT__", vis_script)
        .replace("__GRAPH_JSON__", graph_json)
        .replace("__DOCS_JSON__", docs_json)
        .replace("__TITLE__", title)
    )


def save_plan_graph_json(graph: dict, path: Union[str, Path]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(graph, f, indent=2, default=str)
    return path


def save_plan_graph_dot(graph: dict, path: Union[str, Path]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(plan_graph_to_dot(graph), encoding="utf-8")
    return path


def save_plan_graph_html(
    graph: dict, path: Union[str, Path], inline_vis: Optional[Union[str, Path]] = None
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(plan_graph_to_html(graph, inline_vis=inline_vis), encoding="utf-8")
    return path


def save_plan_graph(
    graph: dict,
    out_dir: Union[str, Path],
    stem: str = "plan_graph",
    render_png: bool = True,
    render_html: bool = True,
    inline_vis: Optional[Union[str, Path]] = None,
) -> Dict[str, Path]:
    """Write ``<stem>.json``, ``<stem>.dot`` and an interactive ``<stem>.html``.

    Also writes ``<stem>.png`` if the Graphviz ``dot`` binary is installed. Pass ``inline_vis``
    (vis-network JS source or a path to it) to make the HTML self-contained / offline-viewable.
    """
    out_dir = Path(out_dir)
    paths = {
        "json": save_plan_graph_json(graph, out_dir / f"{stem}.json"),
        "dot": save_plan_graph_dot(graph, out_dir / f"{stem}.dot"),
    }
    if render_html:
        paths["html"] = save_plan_graph_html(graph, out_dir / f"{stem}.html", inline_vis=inline_vis)
    if render_png:
        png = _try_render_png(paths["dot"], out_dir / f"{stem}.png")
        if png is not None:
            paths["png"] = png
    return paths


def _try_render_png(dot_path: Path, png_path: Path) -> Optional[Path]:
    """Render a DOT file to PNG using the ``dot`` binary if it is available; else skip quietly."""
    import shutil
    import subprocess

    if shutil.which("dot") is None:
        _log.debug("Graphviz 'dot' not found; skipping PNG render of %s", dot_path)
        return None
    try:
        subprocess.run(
            ["dot", "-Tpng", str(dot_path), "-o", str(png_path)],
            check=True,
            capture_output=True,
        )
        return png_path
    except Exception as e:  # pragma: no cover - rendering is best-effort
        _log.warning("Failed to render plan graph PNG: %s", e)
        return None


def to_networkx(graph: dict):  # pragma: no cover - convenience only
    """Convert to a ``networkx.DiGraph`` (requires networkx to be installed)."""
    import networkx as nx

    g = nx.DiGraph(name=graph.get("name"))
    for node in graph["nodes"]:
        attrs = {k: v for k, v in node.items() if k != "id"}
        g.add_node(node["id"], **attrs)
    for edge in graph["edges"]:
        attrs = {k: v for k, v in edge.items() if k not in ("source", "target")}
        g.add_edge(edge["source"], edge["target"], **attrs)
    return g
