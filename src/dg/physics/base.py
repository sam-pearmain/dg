from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, List, Type,  TypeVar, Mapping, Generic, Protocol
from jax import jit
from jaxtyping import Array, Float64

from dg.physics.constants import PhysicalConstant
from dg.physics.variables import StateVector
from dg.physics.interfaces import InterfaceCollection
from dg.utils.pytree import PyTree
from dg.utils.decorators import compose, immutable, debug

@immutable
class PDE(ABC, PyTree):
    """The core PDE abstract base class"""
    def __init__(self, **kwds: PhysicalConstant) -> None:
        for key, value in kwds:
            self.__setattr__(key, value)

    @property
    @abstractmethod
    def state_vector(self) -> StateVector: ...

    @property
    @abstractmethod
    def interface_mapping(self) -> InterfaceCollection: ...

    @property
    @abstractmethod
    def n_dimensions(self) -> int: ...

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
    @staticmethod
    def compute(physics: P, *args: Array) -> Float64[Array, "n_q, n_s"]:
        ...

P = TypeVar('P', bound = PDE)
class NumericalFluxFunction(FluxFunction[P]): # type: ignore
    """A marker trait for numerical fluxes with an additional class variable"""
    _VALID_ON_BOUNDARY: Enum

class ConvectivePDEProtocol(Convective, PDE): 
    """Convective + PDE"""
    ...
    
class DiffusivePDEProtocol(Diffusive, PDE): 
    """Diffusive + PDE"""
    ... 

class ConvectiveDiffusivePDEProtocol(Convective, Diffusive, PDE): 
    """Convective + Diffusive + PDE"""
    ...

C = TypeVar('C', bound = ConvectivePDEProtocol)
class ConvectiveFluxFunction(Generic[C], Convective, FluxFunction): # type: ignore
    """A trait for PDEs with convective analytical flux"""
    @jit
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
    def compute_convective_numerical_flux(
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
    def compute_diffusive_numerical_flux(
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

P = TypeVar('P', bound = PDE, contravariant = True)
N = TypeVar('N', bound = NumericalFluxFunction, contravariant = True)
class NumericalFluxDispatch(Generic[P, N]):
    _flux_branch: Mapping[str, N]
    
    def compute_numerical_flux(
        self, 
    ) -> None:
        pass