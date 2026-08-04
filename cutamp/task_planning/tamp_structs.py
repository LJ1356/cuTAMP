# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from abc import ABC
from dataclasses import dataclass, field
from typing import Sequence, Protocol, List

from cutamp.task_planning import Operator, GroundOperator


# Global cache for interning ground TAMP operators
_GROUND_OP_CACHE: dict[tuple, "GroundTAMPOperator"] = {}


@dataclass(frozen=True, eq=False)
class TAMPOperator(Operator):
    constraints: Sequence["Constraint"] = field(default_factory=tuple)
    costs: Sequence["Cost"] = field(default_factory=tuple)
    # Parameter-name pairs that must bind to DIFFERENT objects. Preconditions cannot express
    # inequality, so without this a two-object operator grounds with both hands on one object.
    # Empty for every single-arm operator, where the search behaviour is therefore unchanged.
    distinct_params: Sequence[tuple] = field(default_factory=tuple)

    def ground(self, substitutions: dict[str, str]) -> "GroundTAMPOperator":
        # Keyed on the operator NAME as well as the substitution: the map alone is ambiguous the
        # moment two operators share a parameter-name set, which would silently return the wrong
        # grounded operator. Today's single-arm operators have pairwise-distinct parameter names, so
        # adding the name cannot split or merge any existing cache entry.
        cache_key = (self.name, tuple(sorted(substitutions.items())))
        if cache_key not in _GROUND_OP_CACHE:
            ground_operator: GroundOperator = super().ground(substitutions)
            # Ground constraints and costs
            ground_constraints = tuple(con.ground(substitutions) for con in self.constraints)
            ground_costs = tuple(cost.ground(substitutions) for cost in self.costs)
            ground_tamp_operator = GroundTAMPOperator(
                **ground_operator.__dict__, constraints=ground_constraints, costs=ground_costs
            )
            _GROUND_OP_CACHE[cache_key] = ground_tamp_operator

        return _GROUND_OP_CACHE[cache_key]


@dataclass(frozen=True)
class GroundTAMPOperator(GroundOperator):
    constraints: Sequence["Constraint"]
    costs: Sequence["Cost"]


PlanSkeleton = List[GroundTAMPOperator]


class Groundable(Protocol):
    def ground(self, substitutions: dict[str, str]) -> "Groundable": ...


class Constraint(ABC, Groundable):
    def __init__(self, *params):
        self.params = params

    @classmethod
    @property
    def type(cls) -> str:
        return cls.__name__

    def ground(self, substitutions: dict[str, str]) -> "Constraint":
        atoms = [substitutions[param.name] for param in self.params]
        return self.__class__(*atoms)

    def __repr__(self):
        name = self.__class__.__name__
        return f"{name}({', '.join(map(str, self.params))})"


class Cost(ABC, Groundable):
    def __init__(self, *params):
        self.params = params

    @classmethod
    @property
    def type(cls) -> str:
        return cls.__name__

    def ground(self, substitutions: dict[str, str]) -> "Cost":
        atoms = [substitutions[param.name] for param in self.params]
        return self.__class__(*atoms)

    def __repr__(self):
        name = self.__class__.__name__
        return f"{name}({', '.join(map(str, self.params))})"
