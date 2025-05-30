from abc import ABC, abstractmethod
from enum import Enum

class Physics(ABC):
    """The physics abstract base class is the skeleton for any weak DG formulation of the governing PDE"""
    def __init__(self):
        super().__init__()

    @abstractmethod
    class StateVariables(Enum):
        pass

    @abstractmethod
    class AdditionalVariables(Enum):
        pass

    