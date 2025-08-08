
from abc import ABC, abstractmethod
from enum import Enum
from jaxtyping import Array, Float64
from typing import Type, Any
from utils import todo

class Physics(ABC):
    """
    The physics abstract base class is the skeleton for any weak DG formulation of a governing 
    advection-diffusion-type PDE
    """
    @property
    @abstractmethod
    def state_variables(self) -> Type[Enum]:
        """The state variables"""
        ...

    @property
    @abstractmethod
    def boundary_conditions(self) -> Type[Enum]:
        """The supported boundary conditions, the meshes cell tags must correspond to these"""
        ...

    @property
    @abstractmethod
    def n_state_vars(self) -> int:
        """Returns the number of state vars of the system"""
        ...

    @property
    @abstractmethod
    def dimensions(self) -> int:
        """Returns the number of physical dimensions of the system"""
        ...

    @abstractmethod
    def conservatives_to_primatives(
        self, 
        u_cons: Float64[Array, "n_q n_s"]
    ) -> Float64[Array, "n_q n_s"]:
        """Computes the primitives from the conservatives"""
        ...

    @abstractmethod
    def primatives_to_conservatives(
        self, 
        u_prim: Float64[Array, "n_q n_s"]
    ) -> Float64[Array, "n_q n_s"]:
        """Computes the conservatives from the primatives"""
        ...

# -- Convective Flux -- 

class ConvectiveFlux(ABC):
    """
    Convective flux terms within the governing equations
    """
    @abstractmethod
    class SupportedConvectiveNumericalFlux(Enum):
        """The supported convective numerical flux functions for a given problem"""
        ...

    @abstractmethod
    def compute_convective_flux(
        self,
        u: Float64[Array, "n_q n_s"]
    ) -> Float64[Array, "n_q n_s"]:
        """Computes the convective flux"""
        ...

    @abstractmethod
    def compute_convective_flux_face(
        self, 
        u: Float64[Array, "n_fq n_s"], 
        normals: Float64[Array, "n_fq n_d"],
    ) -> Float64[Array, "n_fq n_s"]:
        """Computes the convective flux across a face"""
        todo("will likely need to use einsum so we dot with the correct normals")
        ...

    @abstractmethod
    def compute_convective_numerical_flux(
        self,
        u_l: Float64[Array, "n_fq n_s"], 
        u_r: Float64[Array, "n_fq n_s"], 
        normals: Float64[Array, "n_fq n_d"]
    ) -> Float64[Array, "n_fq n_s"]:
        """Computes the convective numerical flux at either inteior or boundary faces"""
        ...

# -- Diffusive Flux --

class DiffusiveFlux(ABC):
    """
    Diffusive flux terms within the governing equations
    """
    @abstractmethod
    def compute_diffusive_flux(
        self,
        u: Float64[Array, "n_q n_s"], 
        grad_u: Float64[Array, "n_q n_s"]
    ) -> Float64[Array, "n_q n_s"]:
        """Computes the diffusive flux within a single element at given quadrature points"""
        ...

    @abstractmethod
    def compute_diffusive_flux_face(
        self, 
        u: Float64[Array, "n_fq n_s"],
        grad_u: Float64[Array, "n_fq n_s"], 
        normals: Float64[Array, "n_fq n_d"],
    ) -> Float64[Array, "n_fq n_s"]:
        """Computes the diffusive flux across a face"""
        ...

    @abstractmethod
    def compute_diffusive_numerical_flux(
        self,
        u_l: Float64[Array, "n_fq n_s"], 
        u_r: Float64[Array, "n_fq n_s"], 
        grad_u_l: Float64[Array, "n_fq n_s"], 
        grad_u_r: Float64[Array, "n_fq n_s"], 
        normals: Float64[Array, "n_fq n_d"]
    ) -> Float64[Array, "n_fq n_s"]:
        """Computes the diffusive numerical flux on interior faces"""
        ... 

    @abstractmethod
    def compute_diffusive_numerical_flux_boundary(
        self,
        u_l: Float64[Array, "n_fq n_s"], 
        u_r: Float64[Array, "n_fq n_s"], 
        grad_u_l: Float64[Array, "n_fq n_s"], 
        normals: Float64[Array, "n_fq n_d"]
    ) -> Float64[Array, "n_fq n_s"]:
        """Computes the diffusive numerical flux on boundary faces"""
        ...

# -- Physical Constants --

class PhysicalConstants:
    """
    The physical constants of the governing equations
    """
    def __init__(self, **kwargs: Any) -> None:
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("Classes which inherit from PhysicalConstants must remain immutable")