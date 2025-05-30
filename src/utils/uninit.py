from typing import Any

class Uninit:
    def __str__(self):
        return "<uninit>"

    def __repr__():
        return "<uninit>"
        
    def __call__(self, *args, **kwds):
        self.uninit()

    def __add__(self, _):
        self.uninit()

    def __radd__(self, _):
        self.uninit()

    def __sub__(self, _):
        self.uninit()

    def __rsub__(self, _):
        self.uninit()

    def __mul__(self, _):
        self.uninit()

    def __rmul__(self, _):
        self.uninit()

    def __truediv__(self, _):
        self.uninit()

    def __rtruediv__(self, _):
        self.uninit()

    def __float__(self):
        self.uninit()

    def __int__(self):
        self.uninit()

    def _uninit():
        raise ValueError("attempted to use uninitialised value")
    
def isuninit(any: Any) -> bool:
    """Just a wrapper for isinstance but to check if something is uninitialised"""
    return isinstance(any, Uninit)