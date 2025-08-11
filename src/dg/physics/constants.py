from typing import Generic, TypeVar

T = TypeVar('T')

class PhysicalConstant(Generic[T]):
    def __init__(self, value: T) -> None:
        self._value = value

    def __get__(self, instance, owner) -> T:
        return self._value

    def __set__(self, instance, owner):
        raise AttributeError("Cannot mutate a PhysicalConstant")

def tests():
    pass

if __name__ == "__main__":
    class Constants:
        a = PhysicalConstant(1.2)
        b = PhysicalConstant(2.4)

    constants = Constants()

    print(1 - constants.a * constants.b)