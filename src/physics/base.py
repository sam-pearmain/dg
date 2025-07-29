import jax.numpy as jnp 

from abc import ABC, abstractmethod, abs
from enum import Enum
from jax import Array

class Physics(ABC):
    """
    The physics abstract base class is the skeleton for any weak DG formulation of a governing 
    advection-diffusion-type PDE
    """
    def __init__(self):
        pass

    @property
    @abstractmethod
    class StateVariables(Enum):
        """The state variables"""
        pass

    @property
    def n_state_vars(self) -> int:
        """Returns the number of state vars of the system"""
        return len(self.StateVariables)

    @property
    @abstractmethod
    def dimensions(self) -> int:
        """The physical dimensions of the system"""
        pass

    @abstractmethod
    def compute_convective_flux(
        self,
        u: Array
    ) -> Array:
        """Computes the analytical convective flux"""
        pass

    @abstractmethod
    def compute_diffusive_flux(
        self,
        u: Array, 
        grad_u: Array
    ) -> Array:
        """Computes the analytical diffusive flux"""
        pass

    @abstractmethod
    def compute_convective_numerical_flux(
        self,
        u_l: Array, 
        u_r: Array, 
        normal: Array
    ) -> Array:
        """Computes the convective numerical flux at either inteior or boundary faces"""
        pass

    @abstractmethod
    def compute_diffusive_numerical_flux(
        self,
        u_l: Array, 
        u_r: Array, 
        grad_u_l: Array, 
        grad_u_r: Array, 
        normal: Array
    ) -> Array:
        """Computes the diffusive numerical flux on interior faces"""
        pass 

    @abstractmethod
    def compute_diffusive_numerical_flux_boundary(
        self,
        u_l: Array, 
        u_r: Array, 
        grad_u_l: Array, 
        normal: Array
    ) -> Array:
        """Computes the diffusive numerical flux on boundary faces"""
        pass

    @abstractmethod
    def compute_source_terms(
        self,
        u: Array, 
        x: Array,
        t: float,
    ) -> Array:
        """Computes the sum of all source terms"""
        pass