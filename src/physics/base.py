import jax.numpy as jnp 

from abc import ABC, abstractmethod
from enum import Enum

class Physics(ABC):
    """The physics abstract base class is the skeleton for any weak DG formulation of the governing PDE"""
    def __init__(self):
        super().__init__()

    @abstractmethod
    class StateVariables(Enum):
        pass

    @property
    def n_state_vars(self):
        """Returns the number of state vars of the system"""
        len(self.StateVariables)

    def initial_condition(self, x):
        """Returns the initial conditions at a given coordinate"""
        return jnp.zeros(self.n_state_vars)