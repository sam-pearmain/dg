from abc import abstractmethod
from enum import Enum
from functools import partial
from typing import Any, Dict, List, Type, Tuple, TypeVar

from jax import jit
from jaxtyping import Array, Float64

from dg.utils.trait import Trait

class PDE(Trait):
    """The core PDE trait"""
    @property
    def n_dimensions(self) -> int: ...

    @property
    def state_variables(self) -> Type[Enum]: ...

    @property
    def n_state_variables(self) -> int: return len(self.state_variables)

    def tree_flatten(self) -> Tuple[List[Any], Dict[str, Any]]: ...

    @classmethod
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

# additional traits
class ConvectivePDE(Convective, PDE, Trait): ...
class DiffusivePDE(Diffusive, PDE, Trait): ... 
class ConvectiveDiffusivePDE(Convective, Diffusive, PDE, Trait): ...

C = TypeVar("C", bound = ConvectivePDE, contravariant = True)
D = TypeVar("D", bound = DiffusivePDE,  contravariant = True)

class ConvectiveNumericalFlux(Trait[C]):
    def compute_convective_numerical_flux(
        self, 
        physics: C, 
        u_l: Float64[Array, "n_fq n_s"], 
        u_r: Float64[Array, "n_fq n_s"],
        normals: Float64[Array, "n_fq n_d"],
    ) -> Float64[Array, "n_fq n_s"]:
        """Computes the convective numerical flux across an element's face"""
        ...

class DiffusiveNumericalFlux(Trait[D]):
    def compute_convective_numerical_flux(
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
