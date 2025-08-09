from abc import ABC, abstractmethod
from jaxtyping import Array, Float64

from dg.physics.base import Physics

# -- convective numerical flux

class ConvectiveNumericalFlux(ABC):
    @abstractmethod
    def compute_convective_numerical_flux(
        self,
        physics: Physics,
        u_l: Float64[Array, "n_fq n_s"],
        u_r: Float64[Array, "n_fq n_s"],
        normals: Float64[Array, "n_fq n_d"]
    ) -> Float64[Array, "n_fq n_s"]:
        """Computes the convective numerical flux at either inteior or boundary faces"""
        ...

# -- diffusive numerical flux --

class DiffusiveNumericalFlux(ABC):
    @abstractmethod
    def compute_diffusive_numerical_flux(
        self,
        u_l: Float64[Array, "n_fq n_s"],
        u_r: Float64[Array, "n_fq n_s"],
        grad_u_l: Float64[Array, "n_fq n_s"],
        grad_u_r: Float64[Array, "n_fq n_s"],
        normals: Float64[Array, "n_fq n_d"]
    ) -> Float64[Array, "n_fq n_s"]:
        """Computes the diffusive numerical flux at either inteior or boundary faces"""
        ...