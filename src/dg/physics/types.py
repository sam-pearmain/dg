from typing import TypeVar, Protocol
from jaxtyping import Float64, Array

class ConvectivePhysics(Protocol):
    def compute_convective_flux(
        self,
        u: Float64[Array, "n_q n_s"]
    ) -> Float64[Array, "n_q n_s"]:
        """Computes the convective flux, F(u)"""
        ...

    def has_convective_terms(self) -> bool:
        return True
