from abc import abstractmethod
from enum import Enum
from functools import partial
from typing import Any, Dict, List, Type, Tuple, TypeVar, Protocol

from jax import jit
from jaxtyping import Array, Float64

from dg.utils.trait import Trait

class PDE(Trait):
    """The core PDE trait"""
    @property
    @abstractmethod
    def n_dimensions(self) -> int: ...

    @property
    @abstractmethod
    def state_variables(self) -> Type[Enum]: ...

    @property
    def n_state_variables(self) -> int: return len(self.state_variables)

    @abstractmethod
    def tree_flatten(self) -> Tuple[List[Any], Dict[str, Any]]: ...

    @classmethod
    @abstractmethod
    def tree_unflatten(
        cls,
        aux_data: Dict[str, Any], 
        children: List[Any]
    ) -> 'PDE': ...

class Convective(Trait):
    """The core trait for convective terms within a PDE"""
    @jit
    @abstractmethod
    def compute_diffusive_flux(
        self,
        u: Float64[Array, "n_q n_s"],
    ) -> Float64[Array, "n_q n_s"]:
        """Computes the convective flux, F_conv(u)"""
        ...

class Diffusive(Trait):
    @abstractmethod
    def compute_diffusive_flux(
        self,
        u: Float64[Array, "n_q n_s"],
        grad_u: Float64[Array, "n_q n_s"],
    ) -> Float64[Array, "n_q n_s"]:
        """Computes the diffusive flux, F_diff(u)"""
        ...

C = TypeVar("C", bound = Convective, contravariant = True)
D = TypeVar("D", bound = Diffusive,  contravariant = True)

class ConvectiveNumericalFlux(Protocol[CPDE]):
    @abstractmethod
    def compute_numerical_flux(
        self, 

    )