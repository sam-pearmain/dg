from typing import Optional, Union, List

from dg.utils.uninit import Uninit

class StateVariable:
    _var_str: str
    _tex_str: Optional[str]
    _idx: Union[int, Uninit]

    def __init__(self, var_name: str, tex_name: Optional[str] = None) -> None:
        self._var_str = var_name
        self._tex_str = tex_name
        self._idx = Uninit()

    def __repr__(self) -> str:
        return self._var_str
    
    def _tex_repr_(self) -> str:
        return self._tex_str if self._tex_str else f"_undefined_"
    
class StateVariables:
    """A container for the state variables"""
    _vars: List[StateVariable]
    _is_zero_indexed: bool = True

    def __init__(self, vars: List[StateVariable]) -> None:
        self._init_var_indexes(vars)
        self._vars = vars

    def __len__(self):
        return len(self._vars)

    @staticmethod
    def _init_var_indexes(vars: List[StateVariable]) -> None:
        for i, variable in enumerate(vars):
            variable._idx = i