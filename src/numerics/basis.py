import jax.numpy as jnp

from enum import Enum, auto
from dataclasses import dataclass
from jax import Array
from numerics.quadrature import QuadratureType
from numerics.quadrature.rules import gauss_lobatto_rule, gauss_legendre_rule
from utils import todo, NotSupportedError

@dataclass(frozen = True)
class BasisData:
    """An immutable container for precomputed basis data on a given reference element and basis function type"""
    elem_type: 'RefElem'
    basis_type: 'BasisType'
    order: int

    # core data
    nodes: Array # the nodal points within the element (n_dofs, dim)
    x_q: Array   # the quadrature points
    w_q: Array   # the quadrature weights 

    @property
    def dimensions(self) -> int:
        return self.elem_type.dimensions()
    
    @property
    def n_dofs(self) -> int:
        return self.nodes.shape[0]

class InterpolationType(Enum):
    Equispaced = auto()
    GaussLobatto = auto()
    GaussLegendre = auto()

class BasisType(Enum):
    Lagrange = auto()

class RefElem(Enum):
    """A reference element enum"""
    Point = auto()
    Line  = auto()
    Quad  = auto()
    Cube  = auto()
    Tri   = auto()
    Tetra = auto()

    def bounds(self) -> tuple[tuple[float, ...], ...]:
        """Returns the bounds of the bounding box for the given reference element"""
        match self:
            case self.Point: return 0
            case self.Line:  return (-1.0, 1.0)
            case self.Quad:  return ((-1.0, -1.0), (1.0, 1.0))
            case self.Cube:  return ((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0))
            case self.Tri:   return ((0.0, 0.0), (1.0, 1.0))
            case self.Tetra: return ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))

    def vertices(self) -> tuple[tuple[float, ...], ...]:
        """Returns the coordinates for each of the vertices of the reference element"""
        match self:
            case self.Point: 
                return ((0.0),)
            case self.Line:  
                return ((-1.0, 1.0))
            case self.Quad:  
                return (
                    (-1.0, -1.0), ( 1.0, -1.0), 
                    ( 1.0,  1.0), (-1.0,  1.0),
                )
            case self.Cube:  
                return (
                    (-1.0, -1.0, -1.0), ( 1.0, -1.0, -1.0),
                    ( 1.0,  1.0, -1.0), (-1.0,  1.0, -1.0),
                    (-1.0, -1.0,  1.0), ( 1.0, -1.0,  1.0),
                    ( 1.0,  1.0,  1.0), (-1.0,  1.0,  1.0),
                )
            case self.Tri:   raise NotImplementedError
            case self.Tetra: raise NotImplementedError

    def dimensions(self) -> int:
        """Returns the number of dimensions for the given reference element"""
        match self: 
            case self.Point: return 0
            case self.Line:  return 1
            case self.Quad:  return 2
            case self.Tri:   return 2
            case self.Cube:  return 3
            case self.Tetra: return 3

    def n_dofs(self, order: int) -> int:
        """Returns the number of degrees of freedom for the reference element"""
        match self:
            case self.Point: return 1
            case self.Line:  return order + 1 
            case self.Quad:  return (order + 1)**2
            case self.Tri:   return (order + 1) * (order + 1) // 2
            case self.Cube:  return (order + 1)**3
            case self.Tetra: return (order + 1) * (order + 2) * (order + 3) // 6

    def get_quadrature_points_weights(self, method: QuadratureType) -> tuple[Array, Array]:
        """Returns the quadrature points and weights for the given reference element and method"""
        match self:
            case self.Point: return (jnp.asarray(0.0, dtype = jnp.float64), jnp.asarray(0.0, dtype = jnp.float64))
            case self.Line:  
                match method:
                    case QuadratureType.GaussLobatto: pass
                    case QuadratureType.GaussLegendre: pass
            case self.Quad: 
                match method:
                    case QuadratureType.GaussLobatto: pass
                    case QuadratureType.GaussLegendre: pass
            case self.Cube: 
                match method:
                    case QuadratureType.GaussLobatto: pass
                    case QuadratureType.GaussLegendre: pass
            case self.Tri:   return 
            case self.Tetra: return 

    def get_interpolation_points(self, order: int, method: InterpolationType) -> Array:
        """Returns the interpolation points over a given element and interpolation type"""
        match self:
            case self.Point: return jnp.asarray(0.0, dtype = jnp.float64)
            case self.Line: 
                match method:
                    case InterpolationType.Equispaced:    return _get_ref_line_equispaced_points(order)
                    case InterpolationType.GaussLobatto:  return _get_ref_line_lobatto_points(order)
                    case InterpolationType.GaussLegendre: return _get_ref_line_legendre_points(order)
            case self.Quad:  
                match method:
                    case InterpolationType.Equispaced:    return _get_ref_quad_equispaced_points(order)
                    case InterpolationType.GaussLobatto:  return _get_ref_quad_lobatto_points(order)
                    case InterpolationType.GaussLegendre: return _get_ref_quad_legendre_points(order)
            case self.Cube:  
                match method: 
                    case InterpolationType.Equispaced:    return _get_ref_cube_equispaced_points(order)
                    case InterpolationType.GaussLobatto:  return _get_ref_cube_lobatto_points(order)
                    case InterpolationType.GaussLegendre: return _get_ref_cube_legendre_points(order)
            case self.Tri:   return 
            case self.Tetra: return

def eval_lagrange_basis(ref_elem: RefElem, x_q: Array) -> Array:
    """Evaluate the Lagrange basis function across a given reference element at points x_q, typically the quadrature points"""
    match ref_elem:
        case RefElem.Point: return 0
        case RefElem.Line:  return _eval_lagrange_ref_line(x_q)
        case RefElem.Quad:  return _eval_lagrange_ref_quad(x_q)
        case RefElem.Cube:  return _eval_lagrange_ref_cube(x_q)
        case RefElem.Tri:   raise NotImplementedError
        case RefElem.Tetra: raise NotImplementedError

# - helpers -

def _construct_1d_lagrange_poly(x_n: Array) -> Array:
    pass

def _eval_lagrange_ref_line(x: Array) -> Array:
    pass

def _eval_lagrange_ref_quad(x: Array) -> Array:
    pass

def _eval_lagrange_ref_cube(x: Array) -> Array:
    pass