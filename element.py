from abc import ABC, abstractmethod
from enum import Enum, auto
import jax
import jax.numpy as jnp

class Dimensions(Enum):
    One   = auto()
    Two   = auto()
    Three = auto()

class ElementType(Enum):
    """An enum encompassing the list of supported element types."""
    Line          = auto()
    Triangle      = auto()
    Quadrilateral = auto()
    Tetrahedra    = auto()
    Hexahedra     = auto()
    Prismatic     = auto()

    def __str__(self):
        ELEMENT_REPR = {
            self.Line:          "line", 
            self.Triangle:      "tri",
            self.Quadrilateral: "quad",
            self.Tetrahedra:    "tetra",
            self.Hexahedra:     "hexa",
        }
        return ELEMENT_REPR[self]

    def dimensions(self):
        """Returns the number of dimensions for the given element."""
        ELEMENT_DIMENSIONS = {
            self.Line:          1, 
            self.Triangle:      2,
            self.Quadrilateral: 2,
            self.Tetrahedra:    3,
            self.Hexahedra:     3,
        }
        return ELEMENT_DIMENSIONS[self]

    def n_nodes(self):
        """Returns the number of nodes corresponding to the given element."""
        ELEMENT_NODE_COUNT = {
            self.Line:          2, 
            self.Triangle:      3,
            self.Quadrilateral: 4,
            self.Tetrahedra:    4,
            self.Hexahedra:     8,
        }
        return ELEMENT_NODE_COUNT[self]
    
    def n_interfaces(self):
        """Returns the number of interfaces corresponding to the given element"""
        ELEMENT_INTERFACE_COUNT = {
            self.Line:       2, 
            self.Triangle:      3,
            self.Quadrilateral: 4,
            self.Tetrahedra:    4,
            self.Hexahedra:     6,
        }
        return ELEMENT_INTERFACE_COUNT[self]

SUPPORTED_ELEMENTS = {
    "line":       ElementType.Line,
    "polygon":    ElementType.Quadrilateral,
    "triangle":   ElementType.Triangle,
    "quad":       ElementType.Quadrilateral,
    "tetra":      ElementType.Tetrahedra,
    "hexahedron": ElementType.Hexahedra,
}