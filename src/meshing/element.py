from typing import Optional
from enum import Enum, auto

class Dimensions(Enum):
    One   = auto()
    Two   = auto()
    Three = auto()

class ElementType(Enum):
    """An enum encompassing the list of supported element types."""
    Point         = auto()
    Line          = auto()
    Triangle      = auto()
    Quadrilateral = auto()
    Tetrahedra    = auto()
    Hexahedra     = auto()

    def __str__(self):
        ELEMENT_REPR = {
            self.Point:         "point",
            self.Line:          "line", 
            self.Triangle:      "tri",
            self.Quadrilateral: "quad",
            self.Tetrahedra:    "tetra",
            self.Hexahedra:     "hexa",
        }
        return ELEMENT_REPR[self]

    def dimensions(self) -> int:
        """Returns the number of dimensions for the given element."""
        ELEMENT_DIMENSIONS = {
            self.Point:         0,
            self.Line:          1, 
            self.Triangle:      2,
            self.Quadrilateral: 2,
            self.Tetrahedra:    3,
            self.Hexahedra:     3,
        }
        return ELEMENT_DIMENSIONS[self]

    def n_nodes(self) -> int:
        """Returns the number of nodes corresponding to the given element."""
        ELEMENT_NODE_COUNT = {
            self.Point:         1,
            self.Line:          2, 
            self.Triangle:      3,
            self.Quadrilateral: 4,
            self.Tetrahedra:    4,
            self.Hexahedra:     8,
        }
        return ELEMENT_NODE_COUNT[self]
    
    def n_interfaces(self) -> int:
        """Returns the number of interfaces corresponding to the given element"""
        ELEMENT_INTERFACE_COUNT = {
            self.Point:         0,
            self.Line:          2, 
            self.Triangle:      3,
            self.Quadrilateral: 4,
            self.Tetrahedra:    4,
            self.Hexahedra:     6,
        }
        return ELEMENT_INTERFACE_COUNT[self]
    
    def face_type(self) -> Optional['ElementType']:
        """Returns the face type of the given element"""
        ELEMENT_FACE_TYPE = {
            self.Point:         None,
            self.Line:          self.Point, 
            self.Triangle:      self.Line,
            self.Quadrilateral: self.Line,
            self.Tetrahedra:    self.Triangle,
            self.Hexahedra:     self.Quadrilateral,
        }
        return ELEMENT_FACE_TYPE[self]
    
    def n_dofs(self, order: int) -> int:
        """Returns the number of degrees of freedom for a given element and its polynomial approximation order"""
        ELEMENT_DOF_FUNCTIONS = {
            self.Point:         lambda p :(1),
            self.Line:          lambda p: (p + 1),
            self.Triangle:      lambda p: (p + 1) * (p + 1) // 2,
            self.Quadrilateral: lambda p: (p + 1)**2,
            self.Tetrahedra:    lambda p: (p + 1) * (p + 2) * (p + 3) // 6,
            self.Hexahedra:     lambda p: (p + 1)**3,
        }
        return ELEMENT_DOF_FUNCTIONS[self](order)

SUPPORTED_ELEMENTS = {
    "point":      ElementType.Point,
    "vertex":     ElementType.Point,
    "line":       ElementType.Line,
    "triangle":   ElementType.Triangle,
    "polygon":    ElementType.Quadrilateral,
    "quad":       ElementType.Quadrilateral,
    "tetra":      ElementType.Tetrahedra,
    "hexahedron": ElementType.Hexahedra,
}