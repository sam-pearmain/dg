from typing import Protocol
from typing import Tuple, List, Dict, Any

class PyTree(Protocol):
    def tree_flatten(self) -> Tuple[List[Any], Dict[str, Any]]: ...
    
    @classmethod
    def tree_unflatten(cls, aux_data: Dict[str, Any], children: List[Any]) -> "PyTree": ...

    def is_pytree(self) -> bool:
        return True