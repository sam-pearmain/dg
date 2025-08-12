from typing import Protocol as Trait
from typing import Tuple, List, Dict, Any

class PyTree(Trait):
    def tree_flatten(self) -> Tuple[List[Any], Dict[str, Any]]: ...
    
    @classmethod
    def tree_unflatten(cls, aux_data: Dict[str, Any], children: List[Any]) -> "PyTree": ...
