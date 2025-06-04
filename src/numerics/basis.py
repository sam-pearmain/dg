import numpy.polynomial

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

class ReferenceElement(Enum):
    """A reference element enum"""
    RefPoint = auto()
    RefLine  = auto()
    RefQuad  = auto()
    RefCube  = auto()
    RefTri   = auto()
    RefTetra = auto()

    def bounds(self) -> tuple[tuple[float, ...], ...]:
        """Returns the bounds of the bounding box for the given reference element"""
        match self:
            case self.RefPoint: return 0
            case self.RefLine:  return (-1.0, 1.0)
            case self.RefQuad:  return ((-1.0, -1.0), (1.0, 1.0))
            case self.RefCube:  return ((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0))
            case self.RefTri:   return ((0.0, 0.0), (1.0, 1.0))
            case self.RefTetra: return ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))

    def dimensions(self) -> int:
        """Returns the number of dimensions for the given reference element"""
        match self: 
            case self.RefPoint: return 0
            case self.RefLine:  return 1
            case self.RefQuad:  return 2
            case self.RefTri:   return 2
            case self.RefCube:  return 3
            case self.RefTetra: return 3

    def n_dofs(self, order: int) -> int:
        """Returns the number of degrees of freedom for the reference element"""
        match self:
            case self.RefPoint: return 1
            case self.RefLine:  return order + 1 
            case self.RefQuad:  return (order + 1)**2
            case self.RefTri:   return (order + 1) * (order + 1) // 2
            case self.RefCube:  return (order + 1)**3
            case self.RefTetra: return (order + 1) * (order + 2) * (order + 3) // 6

    def get_quadrature_points_weights(self) -> tuple[Array, Array]:
        pass

    def get_interpolation_points(self, order: int, method: str) -> Array:
        pass

    def get_equispaced_points(self, order: int) -> Array:
        pass

    def get_gauss_lobatto_points(self, order: int) -> Array:
        pass

    def get_gauss_legendre_points(self, order: int) -> Array:
        pass

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

def _construct_1d_lagrange_poly(x_n: Array) -> n

def _eval_lagrange_ref_line(x: Array) -> Array:
    pass

def _eval_lagrange_ref_quad(x: Array) -> Array:
    pass

def _eval_lagrange_ref_cube(x: Array) -> Array:
    pass