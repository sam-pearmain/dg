from abc import abstractmethod
from enum import Enum
from typing import Any, Dict, List, Type, Tuple, TypeVar, ClassVar

from jax import jit
from jaxtyping import Array, Float64

from dg.physics.constants import PhysicalConstant
from dg.utils.traits import Trait

class PDE(Trait):
    """The core PDE trait"""
    _is_init: bool = False

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
    def state_variables(self) -> Type[Enum]: ...

    @property
    def n_state_variables(self) -> int: return len(self.state_variables)

    @property
    def boundaries(self) -> Type[Enum]: ...

    def get_state_variable_names(self) -> List[str]:
        return list(self.state_variables.__members__.keys())

    def get_state_variable_index(self, var_name: str) -> int:
        return self.get_state_variable_names().index(var_name)

    def tree_flatten(self) -> Tuple[List[Any], Dict[str, Any]]: ...

    @classmethod
    def tree_unflatten(cls, aux_data: Dict[str, Any], children: List[Any]) -> 'PDE': ...

class Convective(Trait):
    def has_convective_terms(self) -> bool: return True

class Diffusive(Trait):
    def has_diffusive_terms(self) -> bool: return True

# trait combinations
class ConvectivePDETrait(Convective, PDE, Trait): ...
class DiffusivePDETrait(Diffusive, PDE, Trait): ... 
class ConvectiveDiffusivePDETrait(Convective, Diffusive, PDE, Trait): ...

C = TypeVar("C", bound = ConvectivePDETrait)
class ConvectiveFlux(Diffusive, Trait[C]): # type: ignore
    """A trait for PDEs with convective analytical flux"""
    @jit
    def compute_diffusive_flux(
        self,
        physics: C, 
        u: Float64[Array, "n_q n_s"],
    ) -> Float64[Array, "n_q n_s"]:
        """Computes the convective flux, F_conv(u)"""
        ...

D = TypeVar("D", bound = DiffusivePDETrait)
class DiffusiveFlux(Convective, Trait[D]): # type: ignore
    @jit
    def compute_diffusive_flux(
        self,
        u: Float64[Array, "n_q n_s"],
        grad_u: Float64[Array, "n_q n_s"],
    ) -> Float64[Array, "n_q n_s"]:
        """Computes the diffusive analytical flux, F_diff(u)"""
        ...

C = TypeVar("C", bound = ConvectivePDETrait)
class ConvectiveNumericalFlux(Trait[C]): # type: ignore
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

D = TypeVar("D", bound = DiffusivePDETrait)
class DiffusiveNumericalFlux(Trait[D]): # type: ignore
    @jit
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
