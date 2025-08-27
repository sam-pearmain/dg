from abc import ABC, abstractmethod
from typing import Generic, TypeVar, Self

from dg.physics.constants import PhysicalConstant
from dg.physics.interfaces import InterfaceType, Interfaces
from dg.physics.variables import StateVector, StateVariable
from dg.physics.flux import Flux
from dg.utils.decorators import compose, autorepr, immutable

P = TypeVar('P', bound = "PDE")
class PDE(ABC, Generic[P]):
    dimensions: int
    state_vector: StateVector
    boundaries: Interfaces[P]
    flux: Flux[P]

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
    def _state_vector_impl(self) -> StateVector[P]: ...

    @abstractmethod
    def _boundaries_impl(self) -> Interfaces[P]: ...

    @abstractmethod
    def _flux_impl(self) -> Flux[P]: ...

def tests():
    class BurgersEquations(PDE["BurgersEquations"]): pass

    class ScalarAdvection(PDE["ScalarAdvection"]):
        def _dimensions_impl(self) -> int:
            return 1
        
        def _state_vector_impl(self) -> StateVector["ScalarAdvection"]:
            return StateVector([
                StateVariable("u"),
            ])
        
        def _boundaries_impl(self) -> Interfaces:
            return Interfaces()
        
        def _flux_impl(self) -> Flux:
            return Flux()
        
    pde = ScalarAdvection()

if __name__ == "__main__":
    tests()