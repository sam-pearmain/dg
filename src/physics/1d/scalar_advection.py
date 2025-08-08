from dataclasses import dataclass
from enum import Enum
from typing import Type
from physics.base import Physics, ConvectiveFlux, PhysicalConstants

@dataclass(frozen = True)
class ScalarAdvectionConstants:
    a: float = 1.0

class ScalarAdvection(Physics, ConvectiveFlux, PhysicalConstants):
    """
    The constant velocity scalar advection equation in 1D.
    
    ∂u/∂t + a * ∂u/∂x = 0
    """
    @property
    def state_variables(self) -> Type[Enum]:
        class StateVariables(Enum):
            u = "u"
        return StateVariables
    
    @property
    def boundary_conditions(self) -> Type[Enum]:
        class BoundaryConditions(Enum):
            Dirichlet = "dirichlet"
            Neumann = "neumann"
        return BoundaryConditions

    @property
    def constants(self) -> ScalarAdvectionConstants:
        return ScalarAdvectionConstants(a = 1.0)
    
