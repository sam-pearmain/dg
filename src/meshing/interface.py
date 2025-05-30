import jax.numpy as jnp

from enum import Enum, auto
from abc import ABC, abstractmethod

class BoundaryType(Enum):
    Wall = auto()
    Farfield = auto()
    VelocityInlet = auto()
    PressureInlet = auto()
    PressureOutlet = auto()

SUPPORTED_BOUNDARIES = {
    "wall": BoundaryType.Wall,
    "farfield": BoundaryType.Farfield,
    "vinlet": BoundaryType.VelocityInlet,
    "pinlet": BoundaryType.PressureInlet,
    "poutlet": BoundaryType.PressureOutlet,
}

class Interface(ABC):
    def __init__(self):
        self.left_element_id = 0
        self.left_element_face_id = 0
        self.right_element_id = 0
        self.right_element_face_id = 0

class Interior(Interface):
    pass

class Boundary(Interface):
    pass

class BoundaryGroup():
    def __init__(self, name: str):
        self.name = name
        self.group_id = 0
        self.boundary_faces = []

    def allocate_boundary_faces(self):
        pass
