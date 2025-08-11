from typing import Protocol, Type, Generic
from jaxtyping import Array, Float64

from dg.physics.interfaces import ConvectivePDEType, DiffusivePDEType

# -- convective numerical flux

class ConvectiveNumericalFlux(Protocol[ConvectivePDEType]):
    def __call__(
        self,
        physics: ConvectivePDEType,
        u_l: Float64[Array, "n_fq n_s"],
        u_r: Float64[Array, "n_fq n_s"],
        normals: Float64[Array, "n_fq n_d"]
    ) -> Float64[Array, "n_fq n_s"]:
        ...

# -- diffusive numerical flux --

class DiffusiveNumericalFlux(Protocol[DiffusivePDEType]):
    def __call__(
        self,
        physics: DiffusivePDEType,
        u_l: Float64[Array, "n_fq n_s"],
        u_r: Float64[Array, "n_fq n_s"],
        grad_u_l: Float64[Array, "n_fq n_s"],
        grad_u_r: Float64[Array, "n_fq n_s"],
        normals: Float64[Array, "n_fq n_d"]
    ) -> Float64[Array, "n_fq n_s"]:
        ...