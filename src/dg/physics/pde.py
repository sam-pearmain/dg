from abc import ABC, abstractmethod
from typing import Protocol, runtime_checkable

from dg.physics.constants import PhysicalConstant
from dg.physics.variables import StateVector, StateVariable
from dg.utils.decorators import compose, autorepr, immutable

class FluxMapping:
    pass

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
    def flux_mapping(self) -> FluxMapping:
        ...

    @property
    def boundaries(self) -> List[Boundaries]: return self.flux_mapping.boundaries()

def tests():
    @runtime_checkable
    class Something(Protocol):
        def true(self) -> bool: ...

    class SomethingElse:
        def true(self) -> bool: return True

    instance = SomethingElse()

    print(isinstance(instance, Something))

if __name__ == "__main__":
    tests()