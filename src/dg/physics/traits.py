from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, List, Type, Tuple, TypeVar, Mapping, Generic, Union

from jax import jit
from jaxtyping import Array, Float64

from dg.physics.constants import PhysicalConstant
from dg.utils.traits import Trait, PyTree
from dg.utils.uninit import Uninit

class PDE(PyTree, Trait):
    """The core PDE trait"""
    _is_init: bool = False

    class StateVariables(Enum):
        ...

    class Boundaries(Enum):
        ...

    def __init__(self, **kwds: PhysicalConstant) -> None:
        for key, value in kwds:
            self.__setattr__(key, value)

        self._is_init = True

    def __setattr__(self, name: str, value: Any) -> None:
        if self._is_init:
            raise AttributeError("once initialised, PDEs remain immutable")
        
        super().__setattr__(name, value)

    @property
    def n_dimensions(self) -> int: ...

    @property
    def state_variables(self) -> Type[StateVariables]: return self.StateVariables

    @property
    def n_state_variables(self) -> int: return len(self.state_variables)

    @property
    def boundaries(self) -> Type[Boundaries]: return self.Boundaries

    def get_state_variable_names(self) -> List[str]:
        return list(self.state_variables.__members__.keys())

    def get_state_variable_index(self, var_name: str) -> int:
        return self.get_state_variable_names().index(var_name)

    def tree_flatten(self) -> Tuple[List[Any], Dict[str, Any]]: ...

    @classmethod
    def tree_unflatten(cls, aux_data: Dict[str, Any], children: List[Any]) -> 'PDE': ...

    def has_convective_terms(self) -> bool: 
        return False

    def has_diffusive_terms(self) -> bool: 
        return False

class Convective(Trait):
    def has_convective_terms(self) -> bool: return True

class Diffusive(Trait):
    def has_diffusive_terms(self) -> bool: return True

P = TypeVar('P', bound = PDE)
class FluxFcn(Trait[P]): # type: ignore
    @jit
    @staticmethod
    def compute(physics: P, *args: Array) -> Float64[Array, "n_q, n_s"]:
        ...

P = TypeVar('P', bound = PDE)
class NumericalFluxFcn(FluxFcn[P]): # type: ignore
    """A marker trait for numerical fluxes with an additional class variable"""
    _VALID_ON_BOUNDARY: Enum

class ConvectivePDETrait(Convective, PDE, Trait): 
    """Convective + PDE"""
    ...
    
class DiffusivePDETrait(Diffusive, PDE, Trait): 
    """Diffusive + PDE"""
    ... 

class ConvectiveDiffusivePDETrait(Convective, Diffusive, PDE, Trait): 
    """Convective + Diffusive + PDE"""
    ...

C = TypeVar('C', bound = ConvectivePDETrait)
class ConvectiveFluxFcn(Convective, FluxFcn, Trait[C]): # type: ignore
    """A trait for PDEs with convective analytical flux"""
    @jit
    def compute(
        self,
        physics: C, 
        u: Float64[Array, "n_q n_s"],
    ) -> Float64[Array, "n_q n_s"]:
        """Computes the convective flux, F_conv(u)"""
        ...

D = TypeVar('D', bound = DiffusivePDETrait)
class DiffusiveFluxFcn(Diffusive, FluxFcn, Trait[D]): # type: ignore
    @jit
    def compute(
        self,
        physics: D, 
        u: Float64[Array, "n_q n_s"],
        grad_u: Float64[Array, "n_q n_s"],
    ) -> Float64[Array, "n_q n_s"]:
        """Computes the diffusive analytical flux, F_diff(u)"""
        ...

C = TypeVar('C', bound = ConvectivePDETrait)
class ConvectiveNumericalFlux(Convective, NumericalFlux, Trait[C]): # type: ignore
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

D = TypeVar('D', bound = DiffusivePDETrait)
class DiffusiveNumericalFlux(Diffusive, NumericalFlux, Trait[D]): # type: ignore
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
N = TypeVar('N', bound = NumericalFluxFcn, contravariant = True)
class NumericalFluxDispatch(Generic[P, N]):
    _flux_branch: Mapping[str, N]
    
    def compute_numerical_flux(
        self, 
        interface: P.boundaries, 
    )