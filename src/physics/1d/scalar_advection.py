from physics import Physics

class ConstantVelocityScalarAdvection1D(Physics):
    """∂u/∂t + a * ∂u/∂x = 0"""
    dimensions = 1

    def __init__(self, a: float):
        self.a = a

    def compute_convective_flux(self, u):
        return u * self.a