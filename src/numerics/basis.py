from abc import ABC, abstractmethod
from enum import Enum, auto
from meshing.element import ElementType
from numerics.quadrature import QuadratureType
from numerics.quadrature.rules import gauss_lobatto_rule, gauss_legendre_rule
from utils.error import *

class BasisType(Enum):
    Lagrange = auto()
    Legendre = auto()

def compute_lagrange_basis(ref_elem: ElementType, order: int, quadrature_type: QuadratureType) -> idk:
    if quadrature_type.is_not_supported_on(ref_elem):
        raise NotSupportedError(f"{quadrature_type} not supported on {ref_elem}")
    
    match quadrature_type:
        case QuadratureType.GaussLobatto:
            quad_nodes, quad_weights = gauss_lobatto_rule(ref_elem, order)
            n_dofs = quad_nodes.shape[0]
        case QuadratureType.GaussLegendre:
            quad_nodes, quad_weights = gauss_legendre_rule(ref_elem, order)
            n_dofs = quad_nodes.shape[0]