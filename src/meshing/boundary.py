from typing import Optional
from enum import Enum, auto

class BoundaryType(Enum):
    Dirichlet = auto()
    Neumann = auto()

SUPPORTED_BOUNDARY_IDS = {
    1: BoundaryType.Dirichlet,
    2: BoundaryType.Neumann
}