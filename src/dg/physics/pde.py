from abc import ABC, abstractmethod
from typing import Generic, Protocol, List, Optional, Type, Any, TypeVar

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
    def flux_mapping(self) -> Type[Flux]:
        ...

    @property
    def interfaces(self) -> List[Interfaces]: return self.flux_mapping.interfaces()

def tests():
    pass

if __name__ == "__main__":
    tests()