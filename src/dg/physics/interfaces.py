from typing import Generic, TypeVar, Mapping, Iterable, TYPE_CHECKING

from dg.utils.decorators import autorepr

if TYPE_CHECKING:
    from dg.physics.pde import PDE

P = TypeVar('P', bound = "PDE")
@autorepr
class InterfaceType(Generic[P]):
    name: str

    def __init__(self, name: str) -> None:
        self.name = name

P = TypeVar('P', bound = "PDE")
@autorepr
class Interfaces(Generic[P]):
    _interfaces: Mapping[int, InterfaceType[P]]

    def __init__(self, interfaces: Mapping[int, InterfaceType[P]]) -> None:
        self._interfaces = interfaces

    def __iter__(self) -> Iterable:
        return self._interfaces.values()

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

if __name__ == "__main__":
    tests()