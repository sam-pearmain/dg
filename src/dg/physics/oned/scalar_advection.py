from enum import Enum
from typing import Any, Type
from dg.physics.base import Physics, ConvectiveFlux, PhysicalConstants

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

def tests():
    from dg.physics.oned import scalar_advection

    system = scalar_advection.ScalarAdvection(a = 1.4)

if __name__ == "__main__":
    tests()
