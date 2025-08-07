
from jaxtyping import Float64
from physics import Physics, ConvectiveTerms

class ConstantVelocityScalarAdvection1D(Physics, ConvectiveTerms):
    """∂u/∂t + a * ∂u/∂x = 0"""
    def __init__(self, a: Float64) -> None:
        super().__init__()

    class StateVariables:
        u = "u"

    class BoundaryTypes:
        DIRICHLET = "dirichlet"
        NEUMANN = "neumann"

    