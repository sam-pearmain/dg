from abc import ABC, abstractmethod
from typing import Generic, TypeVar, Any

from dg.physics.constants import PhysicalConstant
from dg.physics.interfaces import InterfaceType, Interfaces
from dg.physics.variables import StateVector, StateVariable
from dg.physics.flux import Flux
from dg.utils.decorators import compose, autorepr, immutable

P = TypeVar('P', bound = "PDE")
@compose(autorepr, immutable)
class PDE(ABC, Generic[P]):
    _flux: Flux[P]
    _interfaces: Interfaces[P]

    def __init__(self, **kwds: PhysicalConstant) -> None:
        for key, value in kwds.items():
            setattr(self, key, value)
        super().__init__()

    @property
    @abstractmethod
    def n_dimensions(self) -> int: ...

    @property
    @abstractmethod
    def state_vector(self) -> StateVector: ...

    @property
    @abstractmethod
    def interfaces(self) -> Interfaces: ...

    @property
    def flux(self) -> Flux[P]:
        return self._flux

def tests():
    pass

if __name__ == "__main__":
    tests()