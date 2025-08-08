from enum import Enum
from typing import Any, Type
from physics.base import Physics, ConvectiveFlux, PhysicalConstants

class ScalarAdvection(Physics, ConvectiveFlux, PhysicalConstants):
    """
    The constant velocity scalar advection equation in 1D.
    
    ∂u/∂t + a * ∂u/∂x = 0
    """
    def __init__(self, **kwargs: Any) -> None:
        PhysicalConstants.__init__(**kwargs)

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
