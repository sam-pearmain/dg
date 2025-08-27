from abc import ABC, abstractmethod
from typing import Generic, TypeVar, Self

from dg.physics.constants import PhysicalConstant
from dg.physics.interfaces import InterfaceType, Interfaces
from dg.physics.variables import StateVector, StateVariable
from dg.physics.flux import Flux
from dg.utils.decorators import compose, autorepr, immutable

P = TypeVar('P', bound = "PDE")
F = TypeVar('F', bound = "Flux")
class PDE(ABC, Generic[F]):
    dimensions: int
    state_vector: StateVector
    boundaries: Interfaces[Self]
    flux: F

    def __init__(self, **kwds: PhysicalConstant) -> None:
        for key, value in kwds.items():
            setattr(self, key, value)
        
        self.dimensions = self._dimensions_impl()
        self.state_vector = self._state_vector_impl()
        self.boundaries = self._boundaries_impl()
        self.flux = self._flux_impl()

    @abstractmethod
    def _dimensions_impl(self) -> int: ...

    @abstractmethod
    def _state_vector_impl(self) -> StateVector[Self]: ...

    @abstractmethod
    def _boundaries_impl(self) -> Interfaces[Self]: ...

    @abstractmethod
    def _flux_impl(self) -> F: ...

def tests():
    pass 

if __name__ == "__main__":
    tests()