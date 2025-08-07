import jax.numpy as jnp

from typing import Callable
from enum import Enum, auto
from jaxtyping import Float64, Array
from physics import Physics
from utils import todo

class ConvectiveNumericalFluxType(Enum):
    Roe = auto()
    Gudonov = auto()
    HLL = auto()
    HLLC = auto()

    def get_convective_numerical_flux_function(self) -> Callable:
        match self:
            case self.Roe: return roe_flux
            case _: raise NotImplementedError("unknown convective flux function")

def roe_flux(
    u_l: Float64[Array, "n_fq n_s"], 
    u_r: Float64[Array, "n_fq n_s"], 
    normals: Float64[Array, "n_fq n_d"], 
    physics: Physics
) -> Float64[Array, "n_fq n_s"]:
    """
    Computes the convective numerical flux across a face using Roe's approximate Riemann solver
    """
    todo("non-physical behaviour when wave speed is close to zero")