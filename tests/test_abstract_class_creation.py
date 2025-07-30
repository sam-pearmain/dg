from abc import ABC, abstractmethod
from dataclasses import dataclass
from jaxtyping import Float64

class Class1(ABC):
    def __init__(self):
        super().__init__()

    @dataclass(frozen = True)
    @abstractmethod
    class CONSTANTS:
        ...
    
class Class2(Class1):
    def __init__(self):
        super().__init__()
    
    @dataclass(frozen = True)
    class CONSTANTS:
        c: Float64 = 1.0

    @property
    def c(self) -> Float64:
        return self.CONSTANTS.c

class2 = Class2()
print(class2.c)
print(class2.CONSTANTS.c)