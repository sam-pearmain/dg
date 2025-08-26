from abc import ABC, abstractmethod
from typing import Generic, Mapping, Optional, Any, Type, TypeVar, TYPE_CHECKING

from jax import jit
from jaxtyping import Array, Float64

from dg.physics.interfaces import InterfaceType
from dg.utils.decorators import compose, autorepr, immutable

if TYPE_CHECKING:
    from dg.physics.pde import PDE
    from dg.physics.interfaces import InterfaceType, Interfaces

T = TypeVar('T')
class _Function(ABC, Generic[T]): # type: ignore
    @jit
    @abstractmethod
    def __call__(self, *args: Any, **kwds: Any) -> Any: ...

P = TypeVar('P', bound = "PDE")
class _NumericalFluxFunction(_Function[P]):
    @jit
    @abstractmethod
    def __call__(self, physics: P, *args: Any, **kwds: Any) -> Any: ...

    @abstractmethod
    def defined_on(self) -> InterfaceType: ... 

P = TypeVar('P', bound = "PDE")
class ConvectiveAnalyticalFlux(_Function[P]):
    @jit
    @abstractmethod
    def __call__(
            self, 
            physics: P, 
            u: Float64[Array, "n_q n_s"],
        ) -> Float64[Array, "n_q n_s"]:
        ...
    
P = TypeVar('P', bound = "PDE")
class DiffusiveAnalyticalFlux(_Function[P]):
    @jit
    @abstractmethod
    def __call__(
            self, 
            physics: P, 
            u: Float64[Array, "n_q n_s"],
            grad_u: Float64[Array, "n_q n_s n_d"]
        ) -> Float64[Array, "n_q n_s"]:
        ...

P = TypeVar('P', bound = "PDE")
class ConvectiveNumericalFlux(_NumericalFluxFunction[P]):
    @jit
    @abstractmethod
    def __call__(
            self, 
            physics: P, 
            u_l: Float64[Array, "n_q n_s"],
            u_r: Float64[Array, "n_q n_s"], 
            normals: Float64[Array, "n_q n_d"],
        ) -> Float64[Array, "n_q n_s"]:
        ...


P = TypeVar('P', bound = "PDE")
class DiffusiveNumericalFlux(_NumericalFluxFunction[P]):
    @jit
    @abstractmethod
    def __call__(
            self, 
            physics: P, 
            u_l: Float64[Array, "n_q n_s"],
            u_r: Float64[Array, "n_q n_s"], 
            grad_u_l: Float64[Array, "n_q n_s n_d"],
            grad_u_r: Float64[Array, "n_q n_s n_d"],
            normals: Float64[Array, "n_q n_d"],
        ) -> Float64[Array, "n_q n_s"]:
        ...

P = TypeVar('P', bound = "PDE")
N = TypeVar('N', bound = _NumericalFluxFunction)
class _FluxDispatch(ABC, Generic[P, N]):
    _dispatch: Mapping[InterfaceType[P], N]

    @abstractmethod
    def get_numerical_flux_function_on(self, interface: InterfaceType[P]) -> N: 
        return self._dispatch[interface]

P = TypeVar('P', bound = "PDE")
class ConvectiveNumericalFluxDispatch(_FluxDispatch[P, ConvectiveNumericalFlux[P]]):
    def __init__(
            self, 
            mapping: Mapping[InterfaceType[P], ConvectiveNumericalFlux[P]]
        ) -> None:
        self._dispatch = mapping

    def get_numerical_flux_function_on(self, interface: InterfaceType) -> ConvectiveNumericalFlux[P]:
        return super().get_numerical_flux_function_on(interface)

P = TypeVar('P', bound = "PDE")
class DiffusiveNumericalFluxDispatch(_FluxDispatch[P, DiffusiveNumericalFlux[P]]):
    def __init__(
            self,
            mapping: Mapping[InterfaceType[P], DiffusiveNumericalFlux[P]]
        ) -> None:
        self._dispatch = mapping

    def get_numerical_flux_function_on(self, interface: InterfaceType[P]) -> DiffusiveNumericalFlux[P]:
        return super().get_numerical_flux_function_on(interface)

@compose(autorepr, immutable)
class Flux(ABC):
    _convective_analytical_flux:         Optional[ConvectiveAnalyticalFlux]
    _diffusive_analytical_flux:          Optional[DiffusiveAnalyticalFlux]
    _convective_numerical_flux_dispatch: Optional[ConvectiveNumericalFluxDispatch]
    _diffusive_numerical_flux_dispatch:  Optional[DiffusiveNumericalFluxDispatch]

    def __init__(
            self, 
            convective_analytical_flux: Optional[ConvectiveAnalyticalFlux],
            diffusive_analytical_flux:  Optional[DiffusiveAnalyticalFlux], 
            convective_numerical_flux_dispatch:  Optional[ConvectiveNumericalFluxDispatch], 
            diffusive_numerical_flux_dispatch:   Optional[DiffusiveNumericalFluxDispatch],
        ) -> None:
        self._convective_analytical_flux = convective_analytical_flux
        self._diffusive_analytical_flux  = diffusive_analytical_flux
        self._convective_numerical_flux_dispatch = convective_numerical_flux_dispatch
        self._diffusive_numerical_flux_dispatch  = diffusive_numerical_flux_dispatch
        self._sanity_check()
        super().__init__()

    @abstractmethod
    def interfaces(self) -> Interfaces: ...

    @abstractmethod
    def has_convective_terms(self) -> bool: ...
    
    @abstractmethod
    def has_diffusive_terms(self) -> bool: ... 

    def _sanity_check(self) -> None: 
        pass
