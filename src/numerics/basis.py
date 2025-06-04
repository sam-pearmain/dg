from abc import ABC, abstractmethod
from enum import Enum, auto
from jax import Array
from meshing.element import ElementType
from numerics.quadrature import QuadratureType
from numerics.quadrature.rules import gauss_lobatto_rule, gauss_legendre_rule
from utils.error import *

class BasisType(Enum):
    Lagrange = auto()
    Legendre = auto()

def eval_lagrange_basis(ref_elem: ElementType, x_q: Array) -> Array:
    """Evaluate the Lagrange basis function across a given reference element at points x_q, typically the quadrature points"""
    match ref_elem:
        case ElementType.Vertex:        return 0
        case ElementType.Line:          return _eval_lagrange_ref_line(x_q)
        case ElementType.Quadrilateral: return _eval_lagrange_ref_quad(x_q)
        case ElementType.Hexahedra:     return _eval_lagrange_ref_cube(x_q)
        case ElementType.Triangle:      raise NotImplementedError
        case ElementType.Tetrahedra:    raise NotImplementedError

# - helpers -

def _eval_lagrange_ref_line(x: Array) -> Array:
    pass

def _eval_lagrange_ref_quad(x: Array) -> Array:
    pass

def _eval_lagrange_ref_cube(x: Array) -> Array:
    pass