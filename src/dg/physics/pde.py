from abc import ABC, abstractmethod
from typing import Protocol, List, Optional, Type

from dg.physics.constants import PhysicalConstant
from dg.physics.interfaces import Interface, Interfaces
from dg.physics.variables import StateVector, StateVariable
from dg.utils.decorators import compose, autorepr, immutable

@compose(autorepr, immutable)
class Flux(ABC):
    _convective_analytical_flux:         Optional[ConvectiveAnalyticalFluxFunction]
    _diffusive_analytical_flux:          Optional[DiffusiveAnalyticalFluxFunction]
    _convective_numerical_flux_dispatch: Optional[ConvectiveNumericalFluxDispatch]
    _diffusive_numerical_flux_dispatch:  Optional[DiffusiveNumericalFluxDispatch]

    def __init__(
            self, 
            convective_analytical_flux_function: Optional[ConvectiveAnalyticalFluxFunction],
            diffusive_analytical_flux_function:  Optional[DiffusiveAnalyticalFluxFunction], 
            convective_numerical_flux_dispatch:  Optional[ConvectiveNumericalFluxDispatch], 
            diffusive_numerical_flux_dispatch:   Optional[DiffusiveNumericalFluxDispatch],
        ) -> None:
        self._convective_analytical_flux = convective_analytical_flux_function
        self._diffusive_analytical_flux  = diffusive_analytical_flux_function
        self._convective_numerical_flux_dispatch = convective_numerical_flux_dispatch
        self._diffusive_numerical_flux_dispatch  = diffusive_numerical_flux_dispatch
        self._sanity_check()
        super().__init__()

    @abstractmethod
    def interfaces(self) -> Interfaces: ...

    @abstractmethod
    def has_convective_terms(self) -> bool: ...
    
    @abstractmethod
    def has_diffusive_terms(self) -> bool: ... 

    def _sanity_check(self) -> None:
        


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
    @runtime_checkable
    class Something(Protocol):
        def true(self) -> bool: ...

    class SomethingElse:
        def true(self) -> bool: return True

    instance = SomethingElse()

    print(isinstance(instance, Something))

if __name__ == "__main__":
    tests()