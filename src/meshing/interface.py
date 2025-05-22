from abc import ABC, abstractmethod
import jax.numpy as jnp

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
