import sys

from typing import Type, Callable, TypeVar

T = TypeVar('T', bound = object)
def display(debug: bool = False) -> Callable[[Type[T]], Type[T]]:
    """Autogenerates a class' __repr__ method. Note: only looks at instance attributes, not class vars """
    def decorator(cls: Type[T]) -> Type[T]:
        def __repr_standard__(self: T) -> str:
            """Minimal __repr__ method"""
            attrs = ", ".join(f"{key}: {type(value).__name__} = {value!r}" for key, value in self.__dict__.items())
            return f"<{cls.__name__}: {{ {attrs} }}>"

        def __repr_debug__(self: T) -> str:
            """Detailed __repr__ method for debug purposes"""            
            attrs = {key: value for key, value in cls.__dict__.items()
                     if not key.startswith('__') and not callable(value)}
            attrs.update(self.__dict__)
            
            attrs_str = "\n".join(f"\t{key}: {type(value).__name__} = {value!r}"
                                  for key, value in attrs.items())
            
            return (f"{cls.__name__}: {{\n"
                    f"\tMemory: {sys.getsizeof(self)} bytes\n"
                    f"{attrs_str}\n"
                    f"}}")


        if debug:
            setattr(cls, "__repr__", __repr_debug__)
        else:
            setattr(cls, "__repr__", __repr_standard__)

        return cls
    
    return decorator

def tests():
    @display(debug = True)
    class Test():
        def __init__(self) -> None:
            self.one = 1.0
            self.two = 2
            self.str = "str"

    @display(debug = True)
    class Testing():
        def __init__(self) -> None:
            self.test = Test()

    test = Testing()

    print(test)

if __name__ == "__main__":
    tests()