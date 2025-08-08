import numpy as np
import jax.numpy as jnp

from jax import Array
from enum import Enum, auto
from utils import todo
from utils.error import *
from numerics.basis import RefElem

class QuadratureType(Enum):
    GaussLobatto = auto()
    GaussLegendre = auto()

    def __str__(self):
        match self:
            case QuadratureType.GaussLobatto:  return "gauss-lobatto"
            case QuadratureType.GaussLegendre: return "gause-legendre" 

    def is_supported_on(self, ref_elem: RefElem) -> bool:
        """Check whether the given quadrature type is supported on a given reference element"""
        match self:
            case QuadratureType.GaussLobatto: 
                return ref_elem in (
                    RefElem.Point, 
                    RefElem.Line, 
                    RefElem.Quad, 
                    RefElem.Cube,
                )
            case QuadratureType.GaussLegendre:
                return ref_elem in (
                    RefElem.Point, 
                    RefElem.Line, 
                    RefElem.Quad, 
                    RefElem.Cube,
                )
            case _: return False

    def is_not_supported_on(self, ref_elem: RefElem) -> bool:
        return not self.is_supported_on(ref_elem)

def gauss_lobatto_rule(ref_elem: RefElem, order: int) -> tuple[Array, Array]:
    """Computes the Gauss-Lobatto points and weights for a given reference element. These are 
    precomputed at the beginning of the solver and remain unchanged throughout the entire computation"""
    match ref_elem:
        case RefElem.Point: return (jnp.asarray([0.0]), jnp.asarray([0.0]))
        case RefElem.Line:  return _gauss_lobatto_ref_line(order)
        case RefElem.Quad:  return _gauss_lobatto_ref_quad(order)
        case RefElem.Cube:  return _gauss_lobatto_ref_cube(order)
        case RefElem.Tri:   raise NotSupportedError("lobatto quadrature not supported for triangles")
        case RefElem.Tetra: raise NotSupportedError("lobatto quadrature not supported for tetrahedrons")

def gauss_legendre_rule(ref_elem: RefElem, order: int) -> tuple[Array, Array]:
    """Computes the Gauss-Legendre points and weights for a given reference element. These are 
    precomputed at the beginning of the solver and remain unchanged throughout the entire computation"""
    todo("so maybe this is best in a seperate module i'm not sure")
    match ref_elem:
        case RefElem.Point: return (jnp.asarray([0.0]), jnp.asarray([0.0]))
        case RefElem.Line:  return _gauss_legendre_ref_line(order)
        case RefElem.Quad:  return _gauss_legendre_ref_quad(order)
        case RefElem.Cube:  return _gauss_legendre_ref_cube(order)
        case RefElem.Tri:   raise NotSupportedError("gauss-legendre quadrature not supported for triangles")
        case RefElem.Tetra: raise NotSupportedError("gauss-legendre quadrature not supported for tetrahedrons")

# - below is just helper functions -

def _gauss_lobatto_ref_line(order: int) -> tuple[Array, Array]:
    """Computes Gauss-Lobatto points and weights for a reference line element"""
    n = int(np.ceil((order + 3) / 2))

    legendre_poly = np.polynomial.Legendre.basis(n - 1)
    legendre_poly_deriv = legendre_poly.deriv()

    x = np.concatenate(([-1.0, legendre_poly_deriv.roots(), 1.0]))
    w = 2.0 / (n * (n - 1) * legendre_poly(x)**2)
    
    return jnp.asarray(x), jnp.asarray(w)

def _gauss_lobatto_ref_quad(order: int) -> tuple[Array, Array]:
    """Computes Gauss-Lobatto points and weights for a reference quad element"""
    x_1d, w_1d = _gauss_lobatto_ref_line(order)
    
    x1, x2 = jnp.meshgrid(x_1d, x_1d) # x, y
    x = jnp.vstack([x1.ravel(), x2.ravel()]).T
    w = jnp.kron(w_1d, w_1d)

    return x, w

def _gauss_lobatto_ref_cube(order: int) -> tuple[Array, Array]:
    """Computes Gauss-Lobatto points and weights for a reference cube element"""
    x_1d, w_1d = _gauss_lobatto_ref_line(order)
    
    x1, x2, x3 = jnp.meshgrid(x_1d, x_1d, x_1d) # x, y, z
    x = jnp.vstack([x1.ravel(), x2.ravel(), x3.ravel()]).T
    w = jnp.kron(jnp.kron(w_1d, w_1d), w_1d)

    return x, w

def _gauss_legendre_ref_line(order: int) -> tuple[Array, Array]:
    """Computes the Gauss-Legendre points and weights for a reference line element"""
    n = int(np.ceil((order + 1) / 2))

    legendre_poly = np.polynomial.Legendre.basis(n)
    legendre_poly_deriv = legendre_poly.deriv()

    x = legendre_poly_deriv.roots()
    w = 2 / ((1 - x**2) * legendre_poly_deriv(x)**2)

    return jnp.asarray(x), jnp.asarray(w)

def _gauss_legendre_ref_quad(order: int) -> tuple[Array, Array]:
    """Computes the Gauss-Legendre points and weights for a reference quad element"""
    x_1d, w_1d = _gauss_legendre_ref_line(order)
    
    x1, x2 = jnp.meshgrid(x_1d, x_1d) # x, y
    x = jnp.vstack([x1.ravel(), x2.ravel()]).T
    w = jnp.kron(w_1d, w_1d)

    return x, w

def _gauss_legendre_ref_cube(order: int) -> tuple[Array, Array]:
    """Computes the Guass-Legendre points and weights for a reference cube element"""
    x_1d, w_1d = _gauss_legendre_ref_line(order)
    
    x1, x2, x3 = jnp.meshgrid(x_1d, x_1d, x_1d) # x, y, z
    x = jnp.vstack([x1.ravel(), x2.ravel(), x3.ravel()]).T
    w = jnp.kron(jnp.kron(w_1d, w_1d), w_1d)

    return x, w