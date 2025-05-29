import jax.numpy as jnp

from abc import ABC, abstractmethod
from jax import Array

class ConvectiveNumericalFlux(ABC):
    @abstractmethod
    def compute_flux(uq_r: Array, uq_l: Array, normals: Array):
        pass

class DiffusiveNumericalFlux(ABC):
    @abstractmethod
    def compute_flux(uq_r: Array, uq_l: Array, normals: Array):
        pass