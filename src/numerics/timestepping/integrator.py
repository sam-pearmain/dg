import jax.numpy as jnp

from abc import ABC, abstractmethod
from enum import Enum, auto
from solver import Solver

class IntegratorType(Enum):
    """An enum with the supported time integration schemes"""
    ForwardEuler = auto()
    RK4 = auto()

class Integrator(ABC):
    """A abstract class to represent a numerical time integrator"""
    def __init__(self, solver: Solver):
        pass

    @classmethod
    def default(cls, solver: Solver):
        """A default setting for the Integrator class"""
        return ForwardEuler(solver)

    @property
    @abstractmethod
    def type(self) -> IntegratorType:
        """Returns the type of time integration used"""
        pass

    @abstractmethod
    def step(self, solver: Solver):
        """Performs one time step"""
        pass

class ForwardEuler(Integrator):
    type = IntegratorType.ForwardEuler

    def step(self, solver: Solver):
        pass

