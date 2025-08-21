from typing import NewType, Generic, TypeVar, Optional

# a new type just to mark things that should be constant, doesn't actually for immutability though
PhysicalConstant = NewType("PhysicalConstant", float)

T = TypeVar('T')
class Constant(Generic[T]):
    """An enforced immutable constant"""
    _value: T
    _name: Optional[str] = None

    def __init__(self, value: T, name: Optional[str] = None) -> None:
        super().__init__()
        self._value = value
        self._name = name
    
    def __repr__(self) -> str:
        return f"{self._name}: {self._value}" if self._name else f"_undefined_constant_: {self._value}"

    def __get__(self) -> T:
        return self._value
    
    def __set__(self):
        raise ValueError("cannot mutate a constant value")
    
def tests():
    const = Constant(1.2, "gamma")
    print(const)

if __name__ == "__main__":
    tests()