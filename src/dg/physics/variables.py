from typing import Any, Optional, Union, List, Mapping

from dg.utils.uninit import Uninit
from dg.utils.decorators import compose, immutable, debug

@debug
class StateVariable:
    """A single state variable"""
    _var_str: str            # the variable's name
    _tex_str: Optional[str]  # the variable's tex representation, "\rho" for example

    def __init__(self, var_name: str, tex_name: Optional[str] = None) -> None:
        self._var_str = var_name
        self._tex_str = tex_name

    def __repr__(self) -> str:
        return f"StateVariable"
    
    def __str__(self) -> str:
        return self._var_str

    @property
    def name(self) -> str:
        return self._var_str

    @property
    def tex(self) -> str:
        return self._tex_str if self._tex_str else f"_undefined_"
    
@compose(immutable, debug)
class StateVector:
    """A container for the state variables"""
    _vars: List[StateVariable]
    _name_map: Mapping[str, StateVariable]

    def __init__(self, vars: List[StateVariable]) -> None:
        self._vars = vars
        self._name_map = {var._var_str: var for var in vars}

    def __iter__(self):
        return iter(self._vars)

    def __getitem__(self, idx: int) -> StateVariable:
        return self._vars[idx]

    def __getattr__(self, name: str) -> Any:
        if name in self._name_map:
            return self._name_map[name]
        raise AttributeError(f"{type(self).__name__} object has no attribute '{name}'")

    def __len__(self):
        return len(self._vars)
    
    @property
    def n_state_variables(self) -> int: 
        return len(self)

def tests():
    state_vec = StateVector([
        StateVariable("rho", r"\rho"),
        StateVariable("rho_u", r"\rho u"), 
        StateVariable("rho_v", r"\rho v"), 
        StateVariable("e", r"E")
    ])

    print(state_vec.rho)

if __name__ == "__main__":
    tests()