from typing import Any, Optional, Union, List, Dict

from dg.utils.uninit import Uninit

class StateVariable:
    """A single state variable"""
    _var_str: str            # the variable's name
    _tex_str: Optional[str]  # the variable's tex representation, "\rho" for example
    _idx: Union[int, Uninit] # the idx of the variable - relevant for 

    def __init__(self, var_name: str, tex_name: Optional[str] = None) -> None:
        self._var_str = var_name
        self._tex_str = tex_name
        self._idx = Uninit()

    def __repr__(self) -> str:
        return self._var_str
    
    def _tex_repr_(self) -> str:
        return self._tex_str if self._tex_str else f"_undefined_"
    
class StateVector:
    """A container for the state variables"""
    _vars: List[StateVariable]
    _name_map: Dict[str, StateVariable]
    _is_zero_indexed: bool = True

    def __init__(self, vars: List[StateVariable]) -> None:
        self._init_var_indexes(vars)
        self._vars = vars
        self._name_map = {var._var_str: var for var in vars}

    def __iter__(self):
        return iter(self._vars)

    def __getitem__(self, idx: int) -> StateVariable:
        return self._vars[idx]

    def __getattribute__(self, name: str) -> Any:
        if name in self._name_map:
            return self._name_map[name]
        raise AttributeError(f"{type(self).__name__} object has no attribute '{name}'")

    def __len__(self):
        return len(self._vars)

    @staticmethod
    def _init_var_indexes(vars: List[StateVariable]) -> None:
        for i, variable in enumerate(vars):
            variable._idx = i

def tests():
    state_vec = StateVector([
        StateVariable("rho", r"\rho"),
        StateVariable("rho_u", r"\rho u"), 
        StateVariable("rho_v", r"\rho v"), 
        StateVariable("e", r"E")
    ])

    print(state_vec._vars[3]._tex_str)

if __name__ == "__main__":
    tests()