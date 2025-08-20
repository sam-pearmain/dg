from typing import Type, Callable, TypeVar

T = TypeVar('T', bound = object)
def display(debug: bool = False) -> Callable[[Type[T]], Type[T]]:
    """Autogenerates a class' __repr__ method. Note: only looks at instance attributes, not class vars """
    def decorator(cls: Type[T]) -> Type[T]:
        def __repr_standard__(self: T) -> str:
            """Minimal __repr__ method"""
            attrs = ", ".join(f"{key} = {value!r}" for key, value in self.__dict__.items())
            return f"<{cls.__name__}: {{{attrs}}}>"

        def __repr_debug__(self: T) -> str:
            """Detailed __repr__ method for debug purposes"""
            attrs = ", ".join(f"{key}: {type(value).__name__} = {value!r}" for key, value in self.__dict__.items())
            return f"<{cls.__name__}: {{{attrs}}}>"


        if debug:
            setattr(cls, "__repr__", __repr_debug__)
        else:
            setattr(cls, "__repr__", __repr_standard__)

        return cls
    
    return decorator

def tests():
    @display()
    class Test():
        def __init__(self) -> None:
            self.one = 1.0
            self.two = 2
            self.str = "str"

    test = Test()

    print(test)

if __name__ == "__main__":
    tests()