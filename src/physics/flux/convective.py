import jax.numpy as jnp

from enum import Enum, auto
from jax import Array

from physics import Physics

class ConvectiveNumericalFlux(Enum):
    Rusanov = auto()
    Roe = auto()
    HLL = auto()

    def compute_numerical_flux(self, u_l: Array, u_r: Array, normal: Array, physics: Physics) -> Array:
        match self:
            case self.Rusanov: rusanov_flux(u_l, u_r, normal, physics)
            case _: raise NotImplementedError("unknown convective flux function")

def rusanov_flux(u_l: Array, u_r: Array, normal: Array, physics: Physics) -> Array:
    """The Rusanov/local Lax-Friedrichs numerical flux."""
    # get the physical flux from the left and right states
    f_l = physics.convective_flux(u_l, normal)
    f_r = physics.convective_flux(u_r, normal)

    # get the maximum wave speeds (eigenvalues) for the left and right states
    lambda_l = physics.max_eigenvalue(u_l, normal)
    lambda_r = physics.max_eigenvalue(u_r, normal)

    # the maximum wave speed at the interface is the max of the two
    s_max = jnp.maximum(lambda_l, lambda_r)

    # Rusanov flux formula
    return 0.5 * (f_l + f_r) - 0.5 * s_max * (u_r - u_l)