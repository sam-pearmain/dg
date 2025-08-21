from typing import Any, Type, TypeVar, Callable
from functools import wraps

T = TypeVar('T', bound = object)
def compose(*decorators: Callable[[Type[T]], Type[T]]):
    def wrapper(cls: Type[T]) -> Type[T]:
        for decorator in decorators:
            cls = decorator(cls)
        return cls
    return wrapper

T = TypeVar('T', bound = object)
def debug(cls: Type[T]) -> Type[T]:
    """Autogenerates a class' __repr__ method, similar to Rust's [derive(Debug)]"""
    def __repr__(self: T) -> str:
        """Minimal __repr__ method for simple debug"""
        attrs = ", ".join(f"{key}: {type(value).__name__} = {value!r}" for key, value in self.__dict__.items())
        return f"{cls.__name__} {{ {attrs} }}"

    setattr(cls, "__repr__", __repr__)
    return cls

T = TypeVar('T', bound = object)
def immutable(cls: Type[T]) -> Type[T]:
    """Makes a class immutable but not really but kinda"""
    __init__ = cls.__init__
    @wraps(cls.__init__)
    def __new_init__(self: T, *args, **kwds) -> None:
        """Our new __init__"""
        __init__(self, *args, **kwds)
        object.__setattr__(self, "_is_init", True)

    def __new_setattr__(self: T, name: str, value: Any) -> None:
        """Our new __setattr__"""
        if getattr(self, "_is_init", False):
            # if we are initialised
            raise AttributeError(f"immutable object {cls.__name__}")
        else:
            object.__setattr__(self, name, value)
    
    cls.__init__ = __new_init__
    cls.__setattr__ = __new_setattr__
    return cls

def tests():
    @compose(immutable, debug)
    class Point():
        def __init__(self, x, y) -> None:
            self.x = x
            self.y = y

    p1 = Point(1.0, 2.0)

    print(p1)

if __name__ == "__main__":
    tests()