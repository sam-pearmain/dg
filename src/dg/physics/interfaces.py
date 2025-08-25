from typing import Mapping, Iterable

from dg.utils.decorators import autorepr

@autorepr
class Interface:
    name: str

    def __init__(self, name: str) -> None:
        self.name = name

@autorepr
class Interfaces:
    _interfaces: Mapping[int, Interface]

    def __init__(self, interfaces: Mapping[int, Interface]) -> None:
        self._interfaces = interfaces

    def __iter__(self) -> Iterable:
        return self._interfaces.values()

def tests():
    euler_interfaces = Interfaces({
        0: Interface("interior"), 
        1: Interface("pinlet"), 
        2: Interface("poutlet"), 
        3: Interface("wall")
    })

    print(euler_interfaces)

if __name__ == "__main__":
    tests()