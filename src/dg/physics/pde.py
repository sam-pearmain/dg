from abc import ABC, abstractmethod
from typing import Generic, List, Self, TypeVar

from dg.physics.constants import PhysicalConstant
from dg.physics.interfaces import InterfaceType, Interfaces
from dg.physics.variables import StateVector, StateVariable
from dg.physics.flux import Flux
from dg.utils.decorators import compose, autorepr, immutable

@compose(autorepr, immutable)
class PDE(ABC):
    def __init__(self, **kwds: PhysicalConstant) -> None:
        for key, value in kwds.items():
            setattr(self, key, value)
        super().__init__()

    @property
    @abstractmethod
    def state_vector(self) -> StateVector:
        ...

    @property
    @abstractmethod
    def flux(self) -> Flux[Self]:
        ...

    @property
    def interfaces(self) -> Interfaces: 
        return self.flux.interfaces()

def tests():
    from dg.physics.flux import Flux
    class ScalarAdvectionFlux(Flux["ScalarAdvection"]):
        pass
    
    class ScalarAdvection(PDE):
        def __init__(self, **kwds: PhysicalConstant) -> None:
            super().__init__(**kwds)

if __name__ == "__main__":
    tests()