
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from jaxtyping import Array, Float64
from utils import todo

class Physics(ABC):
    """
    The physics abstract base class is the skeleton for any weak DG formulation of a governing 
    advection-diffusion-type PDE
    """
    @abstractmethod
    class StateVariables(Enum):
        """The state variables"""
        ...

    @abstractmethod
    class BoundaryConditions(Enum):
        """The supported boundary conditions for the system"""
        ...

    @abstractmethod
    @dataclass(frozen = True)
    class PhysicalConstants:
        """Container for physical constants"""
        ...

    @property
    def n_state_vars(self) -> int:
        """Returns the number of state vars of the system"""
        return len(self.StateVariables)

    @property
    @abstractmethod
    def dimensions(self) -> int:
        """Returns the number of physical dimensions of the system"""
        ...

    @property
    def constants(self) -> PhysicalConstants:
        """Returns the physical constants"""
        ...

    @abstractmethod
    def conservatives_to_primatives(self, u_cons: Float64[Array, "n_q n_s"]) -> Float64[Array, "n_q n_s"]:
        """Computes the primitives from the conservatives"""
        ...

    @abstractmethod
    def primatives_to_conservatives(self, u_prim: Float64[Array, "n_q n_s"]) -> Float64[Array, "n_q n_s"]:
        """Computes the conservatives from the primatives"""
        ...

class ConvectiveTerms(ABC):
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

class DiffusiveTerms(ABC):
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

class SourceTerms(ABC): 
    @abstractmethod
    def compute_source_terms(
        self,
        u: Array,
        x: Array,
        t: float,
    ) -> Array:
        """Computes the sum of all source terms"""
        ...