from abc import ABC, abstractmethod
from typing import Generic, Mapping, Optional, Any, Type, TypeVar, TYPE_CHECKING

from jax import jit
from jaxtyping import Array, Float64

from dg.physics.interfaces import InterfaceType, Interfaces
from dg.utils.decorators import compose, autorepr, immutable
from dg.utils.todo import todo

if TYPE_CHECKING:
    from dg.physics.pde import PDE

class _Convective:
    """A marker trait for convective terms"""
    def is_convective(self) -> bool: return True

class _Diffusive:
    """A marker trait for diffusive terms"""
    def is_diffusive(self) -> bool: return True

T = TypeVar('T')
class _Function(ABC, Generic[T]): # type: ignore
    """A generic function requiring a jit-compiled __call__ method"""
    @jit
    @abstractmethod
    def __call__(self, *args: Any, **kwds: Any) -> Any: ...

P = TypeVar('P', bound = "PDE")
class _NumericalFluxFunction(_Function[P]):
    @property
    @abstractmethod
    def _DEFINED_ON_INTERFACE(self) -> InterfaceType[P]: ...

    @jit
    @abstractmethod
    def __call__(self, physics: P, *args: Any, **kwds: Any) -> Any: ...

    def defined_on(self) -> InterfaceType[P]: return self._DEFINED_ON_INTERFACE 

P = TypeVar('P', bound = "PDE")
class ConvectiveAnalyticalFlux(_Function[P], _Convective):
    @jit
    @abstractmethod
    def __call__(
            self, 
            physics: P, 
            u: Float64[Array, "n_q n_s"],
        ) -> Float64[Array, "n_q n_s"]:
        ...
    
P = TypeVar('P', bound = "PDE")
class DiffusiveAnalyticalFlux(_Function[P], _Diffusive):
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
    
    def sanity_check(self) -> None:
        if not self._dispatch:
            raise AttributeError("no flux dispatch defined")
        
        for interface_type, numerical_flux_function in self._dispatch.items():
            if numerical_flux_function.defined_on() != interface_type:
                raise AttributeError(
                    f"mismatch interface and numerical flux function: \n"
                    f"  numerical flux: {numerical_flux_function} \n"
                    f"  defined on: {numerical_flux_function.defined_on()} \n"
                    f"  but assigned to {interface_type}"
                )

P = TypeVar('P', bound = "PDE")
class ConvectiveNumericalFluxDispatch(_FluxDispatch[P, ConvectiveNumericalFlux[P]]):
    def __init__(
            self, 
            mapping: Mapping[InterfaceType[P], ConvectiveNumericalFlux[P]]
        ) -> None:
        self._dispatch = mapping
        self.sanity_check()

    def get_numerical_flux_function_on(self, interface: InterfaceType) -> ConvectiveNumericalFlux[P]:
        return super().get_numerical_flux_function_on(interface)

P = TypeVar('P', bound = "PDE")
class DiffusiveNumericalFluxDispatch(_FluxDispatch[P, DiffusiveNumericalFlux[P]]):
    def __init__(
            self,
            mapping: Mapping[InterfaceType[P], DiffusiveNumericalFlux[P]]
        ) -> None:
        self._dispatch = mapping
        self.sanity_check()

    def get_numerical_flux_function_on(self, interface: InterfaceType[P]) -> DiffusiveNumericalFlux[P]:
        return super().get_numerical_flux_function_on(interface)

P = TypeVar('P', bound = "PDE", covariant = True)
@compose(autorepr, immutable)
class Flux(ABC, Generic[P]):
    _convective_analytical_flux:         Optional[ConvectiveAnalyticalFlux[P]]
    _diffusive_analytical_flux:          Optional[DiffusiveAnalyticalFlux[P]]
    _convective_numerical_flux_dispatch: Optional[ConvectiveNumericalFluxDispatch[P]]
    _diffusive_numerical_flux_dispatch:  Optional[DiffusiveNumericalFluxDispatch[P]]

    def __init__(
            self, 
            convective_analytical_flux:         Optional[ConvectiveAnalyticalFlux[P]],
            diffusive_analytical_flux:          Optional[DiffusiveAnalyticalFlux[P]], 
            convective_numerical_flux_dispatch: Optional[ConvectiveNumericalFluxDispatch[P]], 
            diffusive_numerical_flux_dispatch:  Optional[DiffusiveNumericalFluxDispatch[P]],
        ) -> None:
        self._convective_analytical_flux = convective_analytical_flux
        self._diffusive_analytical_flux  = diffusive_analytical_flux
        self._convective_numerical_flux_dispatch = convective_numerical_flux_dispatch
        self._diffusive_numerical_flux_dispatch  = diffusive_numerical_flux_dispatch
        self._sanity_check()

    def interfaces(self) -> Interfaces[P]:
        todo("build this collection when we init the class")

    @abstractmethod
    def has_convective_terms(self) -> bool: ...
    
    @abstractmethod
    def has_diffusive_terms(self) -> bool: ... 

    def _sanity_check(self) -> None: 
        if self.has_convective_terms() and not self._convective_analytical_flux:
            raise AttributeError(
                "flux was marked to have convective terms but convective analytical flux not given"
            )

        if self.has_diffusive_terms() and not self._diffusive_analytical_flux:
            raise AttributeError(
                "flux was marked to have diffusive terms but diffusive analytical flux not given"
            )