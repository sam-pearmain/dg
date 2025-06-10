import jax.numpy as jnp

from enum import Enum, auto
from dataclasses import dataclass
from typing import Optional, Union
from jax import Array
from numerics.quadrature import QuadratureType
from numerics.quadrature.rules import gauss_lobatto_rule, gauss_legendre_rule
from utils import todo, NotSupportedError, Uninit

class InterpolationType(Enum):
    Equispaced = auto()
    GaussLobatto = auto()
    GaussLegendre = auto()

class BasisType(Enum):
    Lagrange = auto()
    Legendre = auto()

@dataclass(frozen = True)
class BasisKey:
    """An immutable container for precomputed basis data on a given reference element and basis function type"""
    elem_type:  'RefElem'
    basis_type:  BasisType
    basis_order: int
    quad_order:  int
    quad_type:   QuadratureType

@dataclass(frozen = True)
class BasisOperators:
    vandermonde: Array # the vandermonde matrix (ij), basis function (j) evaluated at quad point (i)
    derivatives: Array # the partial derivatives of the vandermonde matrix
    
class BasisCache:
    """A cache for all the shape basis functions that lives in RAM"""
    _cache: Union[dict[BasisKey, BasisOperators], Uninit]

    def __init__(self, basis_type: 'BasisType', interpolation: Optional['InterpolationType'] = None):
        self._cache = Uninit

        match basis_type:
            case BasisType.Lagrange: 
                if interpolation is None:
                    raise ValueError("interpolation type must be specified to build lagrange basis functions")
                self.interpolation = interpolation
            case BasisType.Legendre:
                todo()
            case _: NotImplementedError

    def fetch_operators(key: BasisKey) -> BasisOperators:
        pass

    def fetch_vandermonde(key: BasisKey) -> Array:
        pass

    def fetch_derivatives(key: BasisKey) -> Array:
        pass

class QuadratureCache:
    """A cache that stores and indexes quadrature points and weights on demand"""
    cache: dict[tuple['RefElem', QuadratureType, int], tuple[Array, Array]]
    
    def __init__(self):
        self.cache = Uninit

    def fetch_quadrature_points_weights(self, ref_elem: 'RefElem', quad_type: QuadratureType, order: int) -> tuple[Array, Array]:
        pass

    def fetch_quadrature_points(self, ref_elem: 'RefElem', quad_type: QuadratureType, order: int) -> Array:
        pass

    def fetch_quadrature_weights(self, ref_elem: 'RefElem', quad_type: QuadratureType, order: int) -> Array:
        pass

class RefElem(Enum):
    """A reference element enum"""
    Point = auto()
    Line  = auto()
    Quad  = auto()
    Cube  = auto()
    Tri   = auto()
    Tetra = auto()

    def __repr__(self):
        return str(self)

    def __str__(self):
        match self:
            case RefElem.Point: return "point"
            case RefElem.Line:  return "line"
            case RefElem.Quad:  return "quad"
            case RefElem.Cube:  return "cube"
            case RefElem.Tri:   return "triangle"
            case RefElem.Tetra: return "tetrahedron"

    @property
    def bounds(self) -> tuple[tuple[float, ...], ...]:
        """Returns the bounds of the bounding box for the given reference element"""
        match self:
            case self.Point: return 0
            case self.Line:  return (-1.0, 1.0)
            case self.Quad:  return ((-1.0, -1.0), (1.0, 1.0))
            case self.Cube:  return ((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0))
            case self.Tri:   return ((0.0, 0.0), (1.0, 1.0))
            case self.Tetra: return ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))

    @property
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
            case _: raise NotImplementedError(f"not implemented on {self} elements")

    @property
    def dimensions(self) -> int:
        """Returns the number of dimensions for the given reference element"""
        match self: 
            case self.Point: return 0
            case self.Line:  return 1
            case self.Quad:  return 2
            case self.Tri:   return 2
            case self.Cube:  return 3
            case self.Tetra: return 3

    @property
    def n_dofs(self, order: int) -> int:
        """Returns the number of degrees of freedom for the reference element"""
        match self:
            case self.Point: return 1
            case self.Line:  return order + 1 
            case self.Quad:  return (order + 1)**2
            case self.Tri:   return (order + 1) * (order + 1) // 2
            case self.Cube:  return (order + 1)**3
            case self.Tetra: return (order + 1) * (order + 2) * (order + 3) // 6

    def build_basis(self, order: int, ) -> BasisKey:
        """This function precomputes basis data from what it is provided with and returns a BasisKey class"""
        todo()

    def get_quadrature_points_weights(self, order: int, method: QuadratureType) -> tuple[Array, Array]:
        """Returns the quadrature points and weights for the given reference element and method"""
        match method:
            case QuadratureType.GaussLobatto:  return gauss_lobatto_rule(self, order)
            case QuadratureType.GaussLegendre: return gauss_legendre_rule(self, order)

    def get_interpolation_nodes(self, order: int, method: InterpolationType) -> Array:
        """Returns the interpolation nodes over a given element and interpolation type"""
        match self:
            case self.Point: return jnp.asarray(0.0, dtype = jnp.float64)
            case self.Line: 
                match method:
                    case InterpolationType.Equispaced:    return _get_ref_line_equispaced_nodes(order)
                    case InterpolationType.GaussLobatto:  return _get_ref_line_lobatto_nodes(order)
                    case InterpolationType.GaussLegendre: return _get_ref_line_legendre_nodes(order)
            case self.Quad:  
                match method:
                    case InterpolationType.Equispaced:    return _get_ref_quad_equispaced_nodes(order)
                    case InterpolationType.GaussLobatto:  return _get_ref_quad_lobatto_nodes(order)
                    case InterpolationType.GaussLegendre: return _get_ref_quad_legendre_nodes(order)
            case self.Cube:  
                match method: 
                    case InterpolationType.Equispaced:    return _get_ref_cube_equispaced_nodes(order)
                    case InterpolationType.GaussLobatto:  return _get_ref_cube_lobatto_nodes(order)
                    case InterpolationType.GaussLegendre: return _get_ref_cube_legendre_nodes(order)
            case _: raise NotImplementedError(f"not implemeneted on {self} elements")

    def lagrange_vandermonde(self, order: int, interpolation: InterpolationType, quadrature: QuadratureType) -> Array:
        match self:
            case RefElem.Line: return _eval_lagrange_ref_line(order, interpolation, quadrature)
            case RefElem.Quad: return _eval_lagrange_ref_quad(order, interpolation, quadrature)
            case RefElem.Cube: return _eval_lagrange_ref_cube(order, interpolation, quadrature)
            case _: raise NotSupportedError(f"lagrange basis functions not supported on {self}")

    def lagrange_derivative_vandermonde(self, order: int, interpolation: InterpolationType, quadrature: QuadratureType) -> Array:
        match self:
            case RefElem.Line: return _eval_lagrange_derivatives_ref_line(order, interpolation, quadrature)
            case RefElem.Quad: return
            case RefElem.Cube: return

    def legendre_vandermonde(self, order: int) -> Array:
        match self:
            case RefElem.Line: return NotImplementedError
            case RefElem.Quad: return NotImplementedError
            case RefElem.Cube: return NotImplementedError
            case _: raise NotSupportedError(f"legendre basis functions not supported on {self}")

# - interpolation helpers -

def _get_ref_line_equispaced_nodes(order) -> Array:
    n_points = order + 1
    return jnp.linspace(-1, 1, n_points, dtype = jnp.float64)

def _get_ref_quad_equispaced_nodes(order) -> Array: 
    nodes_1d = _get_ref_line_equispaced_nodes(order)
    x, y = jnp.meshgrid(nodes_1d, nodes_1d)
    return jnp.vstack([x.ravel(), y.ravel()]).T

def _get_ref_cube_equispaced_nodes(order) -> Array:
    nodes_1d = _get_ref_line_equispaced_nodes(order)
    x, y, z = jnp.meshgrid(nodes_1d, nodes_1d, nodes_1d)
    return jnp.vstack([x.ravel(), y.ravel(), z.ravel()]).T

def _get_ref_line_lobatto_nodes(order) -> Array:
    nodes, _ = gauss_lobatto_rule(RefElem.Line, order)
    return nodes

def _get_ref_quad_lobatto_nodes(order) -> Array:
    nodes_1d, _ = gauss_lobatto_rule(RefElem.Line, order)
    x, y = jnp.meshgrid(nodes_1d, nodes_1d)
    return jnp.vstack([x.ravel(), y.ravel()]).T

def _get_ref_cube_lobatto_nodes(order) -> Array:
    nodes_1d, _ = gauss_lobatto_rule(RefElem.Line, order)
    x, y, z = jnp.meshgrid(nodes_1d, nodes_1d, nodes_1d)
    return jnp.vstack([x.ravel(), y.ravel(), z.ravel()]).T

def _get_ref_line_legendre_nodes(order) -> Array:
    nodes, _ = gauss_legendre_rule(RefElem.Line, order)
    return nodes

def _get_ref_quad_legendre_nodes(order) -> Array:
    nodes_1d, _ = gauss_legendre_rule(RefElem.Line, order)
    x, y = jnp.meshgrid(nodes_1d, nodes_1d)
    return jnp.vstack([x.ravel(), y.ravel()]).T

def _get_ref_cube_legendre_nodes(order) -> Array:
    nodes_1d, _ = gauss_legendre_rule(RefElem.Line, order)
    x, y, z = jnp.meshgrid(nodes_1d, nodes_1d, nodes_1d)
    return jnp.vstack([x.ravel(), y.ravel(), z.ravel()]).T

# - vandermonde helpers -

def _eval_lagrange_ref_line(order: int, interpolation: InterpolationType, quadrature: QuadratureType) -> Array:
    x_n    = RefElem.Line.get_interpolation_nodes(order, interpolation)
    x_q, _ = RefElem.Line.get_quadrature_points_weights(order, quadrature)

    n_basis_funcs = x_n.shape[0]
    n_quad_points   = x_q.shape[0]

    vandermonde = jnp.ones((n_quad_points, n_basis_funcs), dtype = jnp.float64)

    for i in range (n_basis_funcs):
        for j in range(n_basis_funcs):
            if i == j:
                continue

            numerator = x_q - x_n[j]
            denominator = x_n[i] - x_n[j]

            vandermonde = vandermonde.at[:, i].multiply(numerator / denominator)

    return vandermonde

def _eval_lagrange_ref_quad(order: int, interpolation: InterpolationType, quadrature: QuadratureType) -> Array:
    vandermonde_1d = _eval_lagrange_ref_line(order, interpolation, quadrature)
    return jnp.kron(vandermonde_1d, vandermonde_1d)

def _eval_lagrange_ref_cube(order: int, interpolation: InterpolationType, quadrature: QuadratureType) -> Array:
    vandermonde_1d = _eval_lagrange_ref_line(order, interpolation, quadrature)
    return jnp.kron(jnp.kron(vandermonde_1d, vandermonde_1d), vandermonde_1d)

def _eval_lagrange_derivatives_ref_line(order: int, interpolation: InterpolationType, quadrature: QuadratureType) -> Array:
    x_n    = RefElem.Line.get_interpolation_nodes(order, interpolation)
    x_q, _ = RefElem.Line.get_quadrature_points_weights(order, quadrature)

    n_basis_funcs = x_n.shape[0]
    n_quad_points = x_q.shape[0]

    derivatives = jnp.zeros((n_quad_points, n_basis_funcs), dtype = jnp.float64)

    for i in range(n_basis_funcs):
        for q in range(n_quad_points):
            term_sum = 0
            for k in range(n_basis_funcs):
                if k == i:
                    continue

                term_prod = 1

                for j in range(n_basis_funcs):
                    if j == i or j == k:
                        continue

                    term_prod *= (x_q[q] - x_n[j]) / (x_n[i] - x_n[j])
                
                term_sum += (1 / (x_n[i] - x_n[k])) * term_prod
            
            derivatives = derivatives.at[q, i].set(term_sum)

    return derivatives

def _eval_lagrange_derivatives_ref_quad(order: int, interpolation: InterpolationType, quadrature: QuadratureType) -> Array:
    v_1d = _eval_lagrange_ref_line(order, interpolation, quadrature)
    d_1d = _eval_lagrange_derivatives_ref_line(order, interpolation, quadrature)
    
    dv_dx = jnp.kron(d_1d, v_1d)
    dv_dy = jnp.kron(v_1d, d_1d)

    return jnp.stack([dv_dx, dv_dy], axis = -1)

def _eval_lagrange_derivatives_ref_cube(order: int, interpolation: InterpolationType, quadrature: QuadratureType) -> Array:
    v_1d = _eval_lagrange_ref_line(order, interpolation, quadrature)
    d_1d = _eval_lagrange_derivatives_ref_line(order, interpolation, quadrature)

    dv_dx = jnp.kron(jnp.kron(d_1d, v_1d), v_1d)
    dv_dy = jnp.kron(jnp.kron(v_1d, d_1d), v_1d)
    dv_dz = jnp.kron(jnp.kron(v_1d, v_1d), d_1d)

    return jnp.stack([dv_dx, dv_dy, dv_dz], axis = -1)