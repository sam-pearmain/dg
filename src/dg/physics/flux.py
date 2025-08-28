from abc import ABC, abstractmethod
from typing import overload, Generic, Any, Callable, Literal, TypeVar, ParamSpec, Self, TYPE_CHECKING
from jaxtyping import Array, Float64

if TYPE_CHECKING:
    from dg.physics.pde import PDE


P = ParamSpec('P')
R = TypeVar('R') # the return type
T = TypeVar('T', bound = "_Function")
class _Function(ABC, Generic[P, R]):
    """A stateless, jax-friendly function designed to be namespaced"""
    @abstractmethod
    def __call__(self, *args: Any, **kwds: Any) -> Any: ...        

P = TypeVar('P', bound = "PDE")
class _FluxFunction(_Function, Generic[P]):
    @abstractmethod
    def __call__(self, pde: P, *args: Any, **kwds: Any) -> Any:
        ...

P = TypeVar('P', bound = "PDE")
class _NumericalFluxFunction(_FluxFunction[P]): ...

P = TypeVar('P', bound = "PDE")
class ConvectiveAnalyticalFlux(_FluxFunction[P]): # type: ignore 
    @abstractmethod
    def __call__(
            self, 
            pde: P, 
            u: Float64[Array, "n_q n_s"],
        ) -> Float64[Array, "n_q n_s"]:
        ...
    
P = TypeVar('P', bound = "PDE")
class DiffusiveAnalyticalFlux(Generic[P]):
    @abstractmethod
    def __call__(
            self, 
            pde: P, 
            u: Float64[Array, "n_q n_s"],
            grad_u: Float64[Array, "n_q n_s n_d"]
        ) -> Float64[Array, "n_q n_s"]:
        ...

P = TypeVar('P', bound = "PDE")
class ConvectiveNumericalFlux(_NumericalFluxFunction[P]):
    @abstractmethod
    def __call__(
            self, 
            pde: P, 
            u_l: Float64[Array, "n_q n_s"],
            u_r: Float64[Array, "n_q n_s"], 
            n_vec: Float64[Array, "n_q n_d"],
        ) -> Float64[Array, "n_q n_s"]:
        ...

P = TypeVar('P', bound = "PDE")
class DiffusiveNumericalFlux(_NumericalFluxFunction[P]):
    @abstractmethod
    def __call__(
            self, 
            pde: P, 
            u_l: Float64[Array, "n_q n_s"],
            u_r: Float64[Array, "n_q n_s"], 
            grad_u_l: Float64[Array, "n_q n_s n_d"],
            grad_u_r: Float64[Array, "n_q n_s n_d"],
            n_vec: Float64[Array, "n_q n_d"],
        ) -> Float64[Array, "n_q n_s"]:
        ...

P = TypeVar('P', bound = "PDE")
class Convective(Generic[P]):
    """A convective flux mixin for the lovely autocompletion"""
    _convective_analytical_flux: ConvectiveAnalyticalFlux[P]
    _convective_numerical_flux:  ConvectiveNumericalFlux[P]

    def compute_convective_analytical_flux(
            self, 
            pde: P, 
            u: Float64[Array, "n_q n_s"],
        ) -> Float64[Array, "n_q n_s"]:
        return self._convective_analytical_flux(pde, u)
    
    def compute_convective_numerical_flux(
            self, 
            pde: P, 
            u_l: Float64[Array, "n_q n_s"],
            u_r: Float64[Array, "n_q n_s"], 
            n_vec: Float64[Array, "n_q n_d"]
        ) -> Float64[Array, "n_q n_s"]:
        return self._convective_numerical_flux(pde, u_l, u_r, n_vec)

P = TypeVar('P', bound = "PDE")
class Diffusive(Generic[P]):
    """A diffusive flux mixin for the lovely autocompletion"""
    _diffusive_analytical_flux: DiffusiveAnalyticalFlux[P]
    _diffusive_numerical_flux:  DiffusiveNumericalFlux[P]

    def compute_diffusive_analytical_flux(
            self, 
            pde: P, 
            u: Float64[Array, "n_q n_s"],
            grad_u: Float64[Array, "n_q n_s n_d"],
        ) -> Float64[Array, "n_q n_s"]:
        return self._diffusive_analytical_flux(pde, u, grad_u)
    
    def compute_diffusive_numerical_flux(
            self, 
            pde: P, 
            u_l: Float64[Array, "n_q n_s"],
            u_r: Float64[Array, "n_q n_s"],
            grad_u_l: Float64[Array, "n_q n_s n_d"],
            grad_u_r: Float64[Array, "n_q n_s n_d"], 
            n_vec: Float64[Array, "n_q n_d"]
        ) -> Float64[Array, "n_q n_s"]:
        return self._diffusive_numerical_flux(pde, u_l, u_r, grad_u_l, 
                                              grad_u_r, n_vec)

P = TypeVar('P', bound = "PDE")
class Flux(Generic[P]):
    _convective_analytical_flux: ConvectiveAnalyticalFlux[P] | None
    _convective_numerical_flux:  ConvectiveNumericalFlux[P]  | None
    _diffusive_analytical_flux:  DiffusiveAnalyticalFlux[P]  | None
    _diffusive_numerical_flux:   DiffusiveNumericalFlux[P]   | None

    @overload
    def __init__(
            self, *,
            convective_analytical_flux: ConvectiveAnalyticalFlux[P],
            convective_numerical_flux:  ConvectiveNumericalFlux[P], 
            diffusive_analytical_flux:  Literal[None] = None, 
            diffusive_numerical_flux:   Literal[None] = None,
        ) -> None: ...

    @overload
    def __init__(
            self, *,  
            convective_analytical_flux: Literal[None] = None,
            convective_numerical_flux:  Literal[None] = None, 
            diffusive_analytical_flux:  DiffusiveAnalyticalFlux[P],
            diffusive_numerical_flux:   DiffusiveNumericalFlux[P],
        ) -> None: ...

    @overload
    def __init__(
            self, *, 
            convective_analytical_flux: ConvectiveAnalyticalFlux[P],
            convective_numerical_flux:  ConvectiveNumericalFlux[P], 
            diffusive_analytical_flux:  DiffusiveAnalyticalFlux[P],
            diffusive_numerical_flux:   DiffusiveNumericalFlux[P],
        ) -> None: ...

    def __init__(
        self, *, 
        convective_analytical_flux: ConvectiveAnalyticalFlux[P] | None = None,
        convective_numerical_flux:  ConvectiveNumericalFlux[P]  | None = None, 
        diffusive_analytical_flux:  DiffusiveAnalyticalFlux[P]  | None = None, 
        diffusive_numerical_flux:   DiffusiveNumericalFlux[P]   | None = None,
    ) -> None:
        if convective_numerical_flux and not convective_analytical_flux:
            raise ValueError(
                "cannot have convective numerical flux without convective analytical flux"
            )
        if diffusive_numerical_flux and not diffusive_analytical_flux:
            raise ValueError(
                "cannot have diffusive numerical flux without diffusive analytical flux"
            )
        
        self._convective_analytical_flux = convective_analytical_flux
        self._convective_numerical_flux  = convective_numerical_flux
        self._diffusive_analytical_flux  = diffusive_analytical_flux
        self._diffusive_numerical_flux   = diffusive_numerical_flux
    
    def has_convective_terms(self) -> bool: 
        return True if self._convective_analytical_flux else False
    
    def has_diffusive_terms(self) -> bool: 
        return True if self._diffusive_analytical_flux else False
        
def tests():
    pass

if __name__ == "__main__":
    tests()