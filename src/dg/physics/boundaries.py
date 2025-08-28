from abc import ABC, abstractmethod
from typing import overload, Any, Dict, Generic, TypeVar, Literal, TYPE_CHECKING
from jaxtyping import Array, Float64

if TYPE_CHECKING:
    from dg.physics.pde import PDE

P = TypeVar('P', bound = "PDE")
class Function(ABC, Generic[P]):
    @abstractmethod
    def __call__(self, *args: Any, **kwds: Any) -> Any: ...

P = TypeVar('P', bound = "PDE")
class BoundaryFunction(Function[P]):
    @abstractmethod
    def __call__(self, *args: Any, **kwds: Any) -> Any: ...

P = TypeVar('P', bound = "PDE")
class ConvectiveBoundary(Generic[P]):
    """A mixin for boundary convective flux terms"""
    _u_b_func: BoundaryFunction[P]

    def compute_convective_numerical_flux(
        self,
        pde: P, 
        u_l: Float64[Array, "n_q n_s"],
        normals: Float64[Array, "n_q n_d"],
        *args: Any
    ) -> Float64[Array, "n_q n_s"]:
        u_b = self._u_b_func(args)
        
        return pde.flux.compute_convective_numerical_flux(
            pde, u_l, u_b, normals
        )

P = TypeVar('P', bound = "PDE")
class DiffusiveBoundary(Generic[P]):
    """A mixin for boundary diffusive flux terms"""
    _u_b_func:      BoundaryFunction[P]
    _grad_u_b_func: BoundaryFunction[P]

    def compute_diffusive_numerical_flux(
        self, 
        pde: P,
        u_l: Float64[Array, "n_q n_s"],
        grad_u_l: Float64[Array, "n_q n_s n_d"],
        normals: Float64[Array, "n_q n_d"],
        *args: Any
    ) -> Float64[Array, "n_q n_s"]:
        u_b = self._u_b_func(args)
        grad_u_b = self._grad_u_b_func(pde, u_l, grad_u_l, normals, args)
        
        return pde.flux.compute_diffusive_numerical_flux(
            pde, u_l, u_b, grad_u_l, grad_u_b, normals
        )

class ConvectiveDiffusiveBoundary(Generic[P]):
    """A mixin for boundary diffusive flux terms"""
    _u_b_func:      BoundaryFunction[P]
    _grad_u_b_func: BoundaryFunction[P]

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
            pde, u_l, u_b, grad_u_l, grad_u_b, normals
        )

P = TypeVar('P', bound = "PDE")
class Boundary(Generic[P]):
    _u_b_func:      BoundaryFunction[P] | None
    _grad_u_b_func: BoundaryFunction[P] | None

    @overload
    def __init__(
        self, *, 
        u_b_func:      BoundaryFunction[P], 
        grad_u_b_func: Literal[None] = None,
    ) -> None: ...

    @overload
    def __init__(
        self, *, 
        u_b_func:      BoundaryFunction[P], 
        grad_u_b_func: BoundaryFunction[P],
    ) -> None: ...

    def __init__(
        self, *, 
        u_b_func:      BoundaryFunction[P] | None = None,
        grad_u_b_func: BoundaryFunction[P] | None = None, 
    ) -> None:
        if not u_b_func and not grad_u_b_func:
            raise ValueError(f"boundary conditions not defined")
        
        if grad_u_b_func and not u_b_func:
            raise ValueError(f"for diffusive problems, boundary conditions for u_b and grad_u_b must be defined")
        
        self._u_b_func = u_b_func
        self._grad_u_b_func = grad_u_b_func

    def has_convective_terms(self) -> bool:
        return True if self._u_b_func and not self._grad_u_b_func else False
    
    def has_diffusive_terms(self) -> bool:
        return True if self._u_b_func and self._grad_u_b_func else False

def tests():
    pass

if __name__ == "__main__":
    tests()