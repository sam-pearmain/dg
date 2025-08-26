from abc import ABC, abstractmethod
from typing import Generic, List, Self, TypeVar, Type

from dg.physics.constants import PhysicalConstant
from dg.physics.interfaces import InterfaceType, Interfaces
from dg.physics.variables import StateVector, StateVariable
from dg.physics.flux import Flux
from dg.utils.decorators import compose, autorepr, immutable

P = TypeVar('P', bound = "PDE")
@compose(autorepr, immutable)
class PDE(ABC, Generic[P]):
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
    def flux(self) -> Flux[P]:
        ...

    @property
    def interfaces(self) -> Interfaces: 
        return self.flux.interfaces()

def tests():
    from dg.physics.flux import Flux

    class ScalarAdvectionFlux(Flux["ScalarAdvection"]):
        def has_convective_terms(self) -> bool:
            return True
        
        def has_diffusive_terms(self) -> bool:
            return False
    
    class ScalarAdvection(PDE):
        a: PhysicalConstant = PhysicalConstant(1.0)

        def __init__(self, **kwds: PhysicalConstant) -> None:
            super().__init__(**kwds)

        @property
        def state_vector(self) -> StateVector:
            return StateVector([
                StateVariable("u"),
            ])
        
        @property
        def flux(self) -> ScalarAdvectionFlux:
            return ScalarAdvectionFlux(
                None, 
                None, 
                None, 
                None,
            )
        
    pde = ScalarAdvection()
    print(pde.a)

if __name__ == "__main__":
    tests()