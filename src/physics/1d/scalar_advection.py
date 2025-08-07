from dataclasses import dataclass
from enum import Enum
from physics.base import Physics, ConvectiveFlux, Constants

@dataclass(frozen = True)
class ScalarAdvectionConstants:
    a: float = 1.0

class ScalarAdvection(Physics, ConvectiveFlux, Constants):
    """
    The constant velocity scalar advection equation in 1D.
    
    ∂u/∂t + a * ∂u/∂x = 0
    """
    @property
    def state_variables() -> Enum:
        class StateVariables(Enum):
            u = "u"
        return StateVariables
    
    @property
    def boundary_conditions() -> Enum:
        pass

    @property
    def constants(ScalarAdvectionConstants)
