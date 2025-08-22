from abc import ABC, abstractmethod
from typing import Self, TypeVar, Mapping, Generic, Protocol
from jax import jit
from jaxtyping import Array, Float64

from dg.physics.constants import PhysicalConstant
from dg.physics.variables import StateVector
from dg.physics.interfaces import InterfaceCollection, Interface
from dg.utils.pytree import PyTree
from dg.utils.decorators import compose, immutable, debug


@compose(immutable, debug)
class PDE(ABC, PyTree):
    """The core PDE abstract base class"""
    def __init__(self, **kwds: PhysicalConstant) -> None:
        for key, value in kwds.items():
            self.__setattr__(key, value)
    
    @property
    @abstractmethod
    def n_dimensions(self) -> int: ...

    @property
    @abstractmethod
    def state_vector(self) -> StateVector: ...

    @property
    @abstractmethod
    def interface_mapping(self) -> InterfaceCollection: ...

    @property
    @abstractmethod
    def flux_mapping(self) -> 'FluxMapping[Self]': ...

    def has_convective_terms(self) -> bool: 
        return False

    def has_diffusive_terms(self) -> bool: 
        return False

class Convective(Protocol):
    """Effectively just a marker protocol tbh"""
    def has_convective_terms(self) -> bool: return True

class Diffusive(Protocol):
    """Effectively just a marker protocol tbh"""
    def has_diffusive_terms(self) -> bool: return True

P = TypeVar('P', bound = PDE)
class FluxFunction(Generic[P], Protocol): # type: ignore
    @jit
    @abstractmethod
    def compute(self, physics: P, *args: Array) -> Float64[Array, "n_q, n_s"]:
        ...

P = TypeVar('P', bound = PDE)
class NumericalFluxFunction(FluxFunction[P]): # type: ignore
    """A marker trait for numerical fluxes with an additional class variable"""
    _VALID_ON_INTERFACE: Interface

class ConvectivePDEProtocol(Convective, PDE): 
    """Convective + PDE"""
    ...
    
class DiffusivePDEProtocol(Diffusive, PDE): 
    """Diffusive + PDE"""
    ... 

class ConvectiveDiffusivePDEProtocol(ConvectivePDEProtocol, DiffusivePDEProtocol): 
    """Convective + Diffusive + PDE"""
    ...

C = TypeVar('C', bound = ConvectivePDEProtocol)
class ConvectiveFluxFunction(Generic[C], Convective, FluxFunction): # type: ignore
    """A trait for PDEs with convective analytical flux"""
    @jit
    @abstractmethod
    def compute(
        self,
        physics: C, 
        u: Float64[Array, "n_q n_s"],
    ) -> Float64[Array, "n_q n_s"]:
        """Computes the convective flux, F_conv(u)"""
        ...

D = TypeVar('D', bound = DiffusivePDEProtocol)
class DiffusiveFluxFunction(Generic[D], Diffusive, FluxFunction): # type: ignore
    @jit
    @abstractmethod
    def compute(
        self,
        physics: D, 
        u: Float64[Array, "n_q n_s"],
        grad_u: Float64[Array, "n_q n_s"],
    ) -> Float64[Array, "n_q n_s"]:
        """Computes the diffusive analytical flux, F_diff(u)"""
        ...

C = TypeVar('C', bound = ConvectivePDEProtocol)
class ConvectiveNumericalFluxFunction(Generic[C], Convective, NumericalFluxFunction): # type: ignore
    @jit
    @abstractmethod
    def compute(
        self, 
        physics: C, 
        u_l: Float64[Array, "n_fq n_s"], 
        u_r: Float64[Array, "n_fq n_s"],
        normals: Float64[Array, "n_fq n_d"],
    ) -> Float64[Array, "n_fq n_s"]:
        """Computes the convective numerical flux across an element's face"""
        ...

D = TypeVar('D', bound = DiffusivePDEProtocol)
class DiffusiveNumericalFluxFunction(Generic[D], Diffusive, NumericalFluxFunction): # type: ignore
    @jit
    @abstractmethod
    def compute(
        self, 
        physics: D, 
        u_l: Float64[Array, "n_fq n_s"], 
        u_r: Float64[Array, "n_fq n_s"],
        grad_u_l: Float64[Array, "n_fq n_s"], 
        grad_u_r: Float64[Array, "n_fq n_s"],
        normals: Float64[Array, "n_fq n_d"]
    ) -> Float64[Array, "n_fq n_s"]:
        """Computes the diffusive numerical flux across an element's face"""
        ...

P = TypeVar('P', bound = PDE)
class FluxMapping(Generic[P]): ...

C = TypeVar('C', bound = ConvectivePDEProtocol)
@immutable
class ConvectiveFluxMapping(Convective, FluxMapping[C]): 
    convective_analytical_flux_function: ConvectiveFluxFunction
    convective_numerical_flux_mapping: Mapping[Interface, ConvectiveNumericalFluxFunction[C]]

    def __init__(
            self, 
            analytical_flux: ConvectiveFluxFunction,
            numerical_flux_map: Mapping[Interface, ConvectiveNumericalFluxFunction],
        ) -> None:
        self.convective_analytical_flux_function = analytical_flux
        self.convective_numerical_flux_mapping = numerical_flux_map

D = TypeVar('D', bound = DiffusivePDEProtocol)
@immutable
class DiffusiveFluxMapping(Diffusive, FluxMapping[D]):
    diffusive_analytical_flux_function: DiffusiveFluxFunction
    diffusive_numerical_flux_mapping: Mapping[Interface, DiffusiveNumericalFluxFunction[D]]

    def __init__(
            self, 
            analytical_flux: DiffusiveFluxFunction, 
            numerical_flux_map: Mapping[Interface, DiffusiveNumericalFluxFunction[D]]
        ) -> None:
        self.diffusive_analytical_flux_function = analytical_flux
        self.diffusive_numerical_flux_mapping = numerical_flux_map

E = TypeVar('E', bound = ConvectiveDiffusivePDEProtocol)
@immutable
class ConvectiveDiffusiveFluxMapping(Convective, Diffusive, FluxMapping[E]):
    convective_analytical_flux_function: ConvectiveFluxFunction
    convective_numerical_flux_mapping:   Mapping[Interface, ConvectiveNumericalFluxFunction[E]]
    diffusive_analytical_flux_function:  DiffusiveFluxFunction
    diffusive_numerical_flux_mapping:    Mapping[Interface, DiffusiveNumericalFluxFunction]

    def __init__(
            self,
            convective_analytical_flux:    ConvectiveFluxFunction,
            convective_numerical_flux_map: Mapping[Interface, ConvectiveNumericalFluxFunction],
            diffusive_analytical_flux:     DiffusiveFluxFunction, 
            diffusive_numerical_flux_map:  Mapping[Interface, DiffusiveNumericalFluxFunction],
        ) -> None:
        self.convective_analytical_flux_function = convective_analytical_flux
        self.convective_numerical_flux_mapping   = convective_numerical_flux_map
        self.diffusive_analytical_flux_function  = diffusive_analytical_flux
        self.diffusive_numerical_flux_mapping    = diffusive_numerical_flux_map
        super().__init__()