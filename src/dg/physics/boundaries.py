from abc import ABC, abstractmethod
from typing import Any, Mapping, Generic, TypeVar, TYPE_CHECKING
from jaxtyping import Array, Float64


from dg.physics.flux import _Function

if TYPE_CHECKING:
    from dg.physics.pde import PDE


P = TypeVar('P', bound = "PDE")
class Boundary(ABC, Generic[P]):
    name: str
    _u_b_func: _Function
    _grad_u_b_func: _Function

    def __init__(self, name: str) -> None:
        self.name = name.lower()

    def compute_convective_numerical_flux(
        self,         
        pde: P, 
        u_l: Float64[Array, "n_q n_s"],
        normals: Float64[Array, "n_q n_d"],
        *args: Any
    ) -> Float64[Array, "n_q n_s"]:
        u_b = self._u_b_func(pde, u_l, normals, args)
        return pde.flux.compute_convective_numerical_flux(
            pde, u_l, u_b, normals
        )

    def compute_diffusive_numerical_flux(
        self, 
        pde: P,
        u_l: Float64[Array, "n_q n_s"],
        grad_u_l: Float64[Array, "n_q n_s n_d"],
        normals: Float64[Array, "n_q n_d"],
        *args: Any
    ) -> Float64[Array, "n_q n_s"]:
        u_b = self._u_b_func(pde, u_l, normals, args)
        grad_u_b = self._grad_u_b_func(pde, u_l, grad_u_l, normals, args)
        return pde.flux.compute_diffusive_numerical_flux(
            pde, u_l, u_b, grad_u_l, grad_u_b, normals, args
        )

    @abstractmethod
    def _u_func_impl(self, *args: Any) -> _Function: ...

    @abstractmethod
    def _grad_u_func_impl(self, *args: Any) -> _Function: ...

class BoundaryCollection(Generic[P]):
    _collection: Mapping[int, Boundary]