from typing import Self, Generic, TypeVar, Mapping, Iterable, TYPE_CHECKING

from dg.utils.decorators import autorepr

if TYPE_CHECKING:
    from dg.physics.pde import PDE

P = TypeVar('P', bound = "PDE")
class InterfaceType(Generic[P]):
    name: str

    def __init__(self, name: str) -> None:
        self.name = name.lower()

    def is_interior(self) -> bool:
        return True if self.name == "interior" else False

P = TypeVar('P', bound = "PDE")
class Interfaces(Generic[P]):
    _interfaces: Mapping[int, InterfaceType[P]]

    def __init__(self, interfaces: Mapping[int, InterfaceType[P]]) -> None:
        self._interfaces = interfaces

    def __iter__(self) -> Iterable:
        return self._interfaces.values()
    
    @property
    def boundaries(self) -> Self:
        boundaries = {
            id: interface
            for id, interface in self._interfaces.items()
            if not interface.is_interior()
        }
        return type(self)(boundaries)

def tests():
    from dg.physics.pde import PDE

    class Euler(PDE):
        pass

    euler_interfaces = Interfaces[Euler]({
        0: InterfaceType[Euler]("interior"), 
        1: InterfaceType[Euler]("pinlet"), 
        2: InterfaceType[Euler]("poutlet"), 
        3: InterfaceType[Euler]("wall")
    })

    print(euler_interfaces)
    print(euler_interfaces.boundaries)

if __name__ == "__main__":
    tests()