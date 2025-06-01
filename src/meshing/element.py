from typing import Optional
from enum import Enum, auto

SUPPORTED_ELEMENTS = [
    "point", "vertex", 
    "line",
    "triangle", "tri",
    "quadrilateral", "quad",
    "tetrahedron", "tetra",
    "hexahedron", "hexa",
]

class ElementType(Enum):
    """An enum encompassing the list of supported element types."""
    Vertex        = auto()
    Line          = auto()
    Triangle      = auto()
    Quadrilateral = auto()
    Tetrahedra    = auto()
    Hexahedra     = auto()

    def __str__(self):
        # note: these str representations correspond to meshio's element names
        return {
            self.Vertex:        "vertex",
            self.Line:          "line", 
            self.Triangle:      "triangle",
            self.Quadrilateral: "quad",
            self.Tetrahedra:    "tetra",
            self.Hexahedra:     "hexahedron",
        }[self]
    
    @classmethod
    def from_str(cls, s: str) -> 'ElementType':
        """Returns the corresponding ElementType for the given input string"""
        if s not in SUPPORTED_ELEMENTS:
            raise NotImplementedError(f"'{s}' elements not supported")
        return {
            "point":      cls.Vertex, 
            "vertex":     cls.Vertex,
            "line":       cls.Line,
            "triangle":   cls.Triangle,
            "polygon":    cls.Quadrilateral,
            "quad":       cls.Quadrilateral,
            "tetra":      cls.Tetrahedra,
            "hexahedron": cls.Hexahedra,
        }[s]
        
    @staticmethod
    def is_supported(s: str) -> bool:
        """Just a quick static method to check whether a given str is a supported element type"""
        return True if s in SUPPORTED_ELEMENTS else False

    def as_meshio_str(self) -> str:
        """Returns the meshio str representation for the given element"""
        return str(self)

    @property
    def dimensions(self) -> int:
        """Returns the number of dimensions for the given element."""
        return {
            self.Vertex:        0,
            self.Line:          1, 
            self.Triangle:      2,
            self.Quadrilateral: 2,
            self.Tetrahedra:    3,
            self.Hexahedra:     3,
        }[self]

    @property
    def n_nodes(self) -> int:
        """Returns the number of nodes corresponding to the given element."""
        return {
            self.Vertex:        1,
            self.Line:          2, 
            self.Triangle:      3,
            self.Quadrilateral: 4,
            self.Tetrahedra:    4,
            self.Hexahedra:     8,
        }[self]
    
    @property
    def n_interfaces(self) -> int:
        """Returns the number of interfaces corresponding to the given element"""
        return {
            self.Vertex:        0,
            self.Line:          2, 
            self.Triangle:      3,
            self.Quadrilateral: 4,
            self.Tetrahedra:    4,
            self.Hexahedra:     6,
        }[self]
    
    @property
    def face_type(self) -> Optional['ElementType']:
        """Returns the face type of the given element"""
        return {
            self.Vertex:        None,
            self.Line:          self.Vertex, 
            self.Triangle:      self.Line,
            self.Quadrilateral: self.Line,
            self.Tetrahedra:    self.Triangle,
            self.Hexahedra:     self.Quadrilateral,
        }[self]
    
    @property
    def n_dofs(self, order: int) -> int:
        """Returns the number of degrees of freedom for a given element and its polynomial approximation order"""
        return {
            self.Vertex:        lambda p :(1),
            self.Line:          lambda p: (p + 1),
            self.Triangle:      lambda p: (p + 1) * (p + 1) // 2,
            self.Quadrilateral: lambda p: (p + 1)**2,
            self.Tetrahedra:    lambda p: (p + 1) * (p + 2) * (p + 3) // 6,
            self.Hexahedra:     lambda p: (p + 1)**3,
        }[self](order)