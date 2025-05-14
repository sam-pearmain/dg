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
    Segment       = auto()
    Triangle      = auto()
    Quadrilateral = auto()
    Tetrahedra    = auto()
    Hexahedra     = auto()
    Prismatic     = auto()

    def __str__(self):
        ELEMENT_REPR = {
            ElementType.Segment:       "seg", 
            ElementType.Triangle:      "tri",
            ElementType.Quadrilateral: "quad",
            ElementType.Tetrahedra:    "tetra",
            ElementType.Hexahedra:     "hexa",
        }
        return ELEMENT_REPR[self]

    def dimensions(self):
        """Returns the number of dimensions for the given element."""
        ELEMENT_DIMENSIONS = {
            ElementType.Segment:       1, 
            ElementType.Triangle:      2,
            ElementType.Quadrilateral: 2,
            ElementType.Tetrahedra:    3,
            ElementType.Hexahedra:     3,
        }
        return ELEMENT_DIMENSIONS[self]

    def n_nodes(self):
        """Returns the number of nodes corresponding to the given element."""
        ELEMENT_NODE_COUNT = {
            ElementType.Segment:       2, 
            ElementType.Triangle:      3,
            ElementType.Quadrilateral: 4,
            ElementType.Tetrahedra:    4,
            ElementType.Hexahedra:     8,
        }
        return ELEMENT_NODE_COUNT[self]
    
    def n_interfaces(self):
        """Returns the number of interfaces corresponding to the given element"""
        ELEMENT_INTERFACE_COUNT = {
            ElementType.Segment:       2, 
            ElementType.Triangle:      3,
            ElementType.Quadrilateral: 4,
            ElementType.Tetrahedra:    4,
            ElementType.Hexahedra:     6,
        }
        return ELEMENT_INTERFACE_COUNT[self]

class Element(ABC):
    """The most fundemental finite element class"""
    def __init__(self, id: int, element_type: ElementType):
        self.id = id
        self.type = element_type
        self.node_ids = jnp.zeros(element_type.n_nodes(), dtype = int)
        
        self.neighbours = []
        self.interfaces = []

    def __repr__(self):
        return f"Element(id = {self.id}, type = {self.type}, node_ids: {self.node_ids})"

class Segement(Element):
    def __init__(self, id, nodes_ids: list, element_type = ElementType.Segment):
        super().__init__(id, element_type)

        if nodes_ids.len() != 2:
            raise TypeError("invalid number of nodes for a seg element")

        self.node_ids = nodes_ids
        self.neighbours = jnp.zeros(2, dtype = int)

class Triangle(Element):
    def __init__(self, id, nodes_ids: list, element_type = ElementType.Triangle):
        super().__init__(id, element_type)

        if nodes_ids.len() != 3:
            raise TypeError("invalid number of nodes for a tri element")

        self.node_ids = nodes_ids
        self.neighbours = jnp.zeros(2, dtype = int)

class Quadrilateral(Element):
    def __init__(self, id, nodes_ids: list, element_type = ElementType.Quadrilateral):
        super().__init__(id, element_type)

        if nodes_ids.len() != 4:
            raise TypeError("invalid number of nodes for a quad element")

        self.node_ids = nodes_ids
        self.neighbours = jnp.zeros(2, dtype = int)