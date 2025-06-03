from abc import ABC, abstractmethod
from enum import Enum, auto
from meshing.element import ElementType

class BasisType(Enum):
    Lagrange = auto()
    Legendre = auto()

def compute_lagrange_basis(ref_elem: ElementType)