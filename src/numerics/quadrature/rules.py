import numpy as np
import jax.numpy as jnp

from jax import Array
from utils import todo
from utils.error import *
from meshing.element import ElementType

def gauss_lobatto_rule(ref_elem: ElementType, order: int) -> tuple[Array, Array]:
    """Computes the Gauss-Lobatto points and weights for a given reference element. These are 
    precomputed at the beginning of the solver and remain unchanged throughout the entire computation"""
    match ref_elem:
        case ElementType.Vertex:        return (jnp.asarray([0.0]), jnp.asarray([0.0]))
        case ElementType.Line:          return _gauss_lobatto_ref_line(order)
        case ElementType.Quadrilateral: return _gauss_lobatto_ref_quad(order)
        case ElementType.Hexahedra:     return _gauss_lobatto_ref_cube(order)
        case ElementType.Triangle:      raise NotSupportedError("lobatto quadrature not supported for triangles")
        case ElementType.Tetrahedra:    raise NotSupportedError("lobatto quadrature not supported for tetrahedra")

def gauss_legendre_rule(ref_elem: ElementType, order: int) -> tuple[Array, Array]:
    todo("so maybe this is best in a seperate module i'm not sure")
    match ref_elem:
        case ElementType.Vertex:        return (jnp.asarray([0.0]), jnp.asarray([0.0]))
        case ElementType.Line:          return _gauss_legendre_ref_line(order)
        case ElementType.Quadrilateral: return _gauss_legendre_ref_quad(order)
        case ElementType.Hexahedra:     return _gauss_legendre_ref_cube(order)
        case ElementType.Triangle:      raise NotSupportedError("gauss-legendre quadrature not supported for triangles")
        case ElementType.Tetrahedra:    raise NotSupportedError("gauss-legendre quadrature not supported for tetrahedra")

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
    n = x_1d.shape[0]

    # perform the tensor product
    x, w = [], []
    for i in range(n):
        for j in range(n):
            x.append([x_1d[i], x_1d[j]])
            w.append(w_1d[i] * w_1d[j])

    return jnp.asarray(x), jnp.asarray(w)

def _gauss_lobatto_ref_cube(order: int) -> tuple[Array, Array]:
    """Computes Gauss-Lobatto points and weights for a reference cube element"""
    x_1d, w_1d = _gauss_lobatto_ref_line(order)
    n = x_1d.shape[0]

    # perform the tensor product
    x, w = [], []
    for i in range(n):
        for j in range(n):
            for k in range(n):
                x.append([x_1d[i], x_1d[j], x_1d[k]])
                w.append(w_1d[i] * w_1d[j] * w_1d[k])

    return jnp.asarray(x), jnp.asarray(w)

def _gauss_legendre_ref_line(order: int) -> tuple[Array, Array]:
    """Computes the Gauss-Legendre points and weights for a reference line element"""
    n = int(np.ceil((order + 1) / 2))

    legendre_poly = np.polynomial.Legendre.basis(n)
    legendre_poly_deriv = legendre_poly.deriv()

    x = np.concatenate([-1.0, legendre_poly_deriv.roots(), 1.0])
    w = 2 / ((1 - x**2) * legendre_poly_deriv(x)**2)

    return x, w

def _gauss_legendre_ref_quad(order: int) -> tuple[Array, Array]:
    """Computes the Gauss-Legendre points and weights for a reference quad element"""
    x_1d, w_1d = _gauss_legendre_ref_line(order)
    n = x_1d.shape[0]

    # perform the tensor product
    x, w = [], []
    for i in range(n):
        for j in range(n):
            x.append([x_1d[i], x_1d[j]])
            w.append(w_1d[i] * w_1d[j])

    return jnp.asarray(x), jnp.asarray(w)

def _gauss_legendre_ref_cube(order: int) -> tuple[Array, Array]:
    """Computes the Guass-Legendre points and weights for a reference cube element"""
    x_1d, w_1d = _gauss_legendre_ref_line(order)
    n = x_1d.shape[0]

    # perform the tensor product
    x, w = [], []
    for i in range(n):
        for j in range(n):
            for k in range(n):
                x.append([x_1d[i], x_1d[j], x_1d[k]])
                w.append(w_1d[i] * w_1d[j] * w_1d[k])

    return jnp.asarray(x), jnp.asarray(w)