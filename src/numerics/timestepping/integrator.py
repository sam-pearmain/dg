import jax.numpy as jnp

from abc import ABC, abstractmethod
from enum import Enum, auto

class IntegratorType(Enum):
    ForwardEuler = auto()
    RK4 = auto()

class Integrator(ABC):
    def __init__(self, u):
        self.residuals = jnp.zeros_like(u)
        self.dt = 0.
        self.n_steps = 0
        self.get_time_step_size = None

    @property
    @abstractmethod
    def type(self) -> IntegratorType:
        pass

    @abstractmethod
    def step(self, solver):
        pass

class ForwardEuler(Integrator):
    type = IntegratorType.ForwardEuler

    def step(self, solver):
        pass

