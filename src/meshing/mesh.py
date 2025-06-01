import jax.numpy as jnp
import numpy as np
import meshio

from jax import Array
from numpy.typing import ArrayLike
from typing import Optional, Dict, Union
from warnings import warn    
from meshio import Mesh as MeshIOMesh
from utils import todo, Uninit, MeshReadError
from meshing.boundary import SUPPORTED_BOUNDARY_IDS
from meshing.element import ElementType, SUPPORTED_ELEMENTS

class ElementConnectivity:
    """Element connectivities stored in JAX arrays."""
    def __init__(self, elements: Array, boundary_faces: Array):
        self.elements = elements
        self.boundary_faces = boundary_faces

    def cull_non_boundary_faces(self, boundary_entity_ids: ArrayLike) -> Array:
        """This function takes the array storing all the boundary entity 
        ids and deletes those which are undefined, i.e., id = -1."""
        ids = jnp.asarray(boundary_entity_ids, dtype = jnp.int32).reshape(-1)
        allowed = jnp.asarray(list(SUPPORTED_BOUNDARY_IDS.keys()), dtype = jnp.int32)
        
        # create the boolean mask (not -1 & is in supported boundary ids)
        mask = (ids != -1) & jnp.isin(ids, allowed)
        
        # warn if we encounter unsupported boundaries (not -1, but not supported)
        bad = jnp.unique(ids[(ids != -1) & ~jnp.isin(ids, allowed)])
        for b in bad:
            warn(f"unsupported boundary id {int(b)}, skipping")

        kept_faces = self.boundary_faces[mask]
        kept_ids   = ids[mask]

        if kept_faces.shape[0] != kept_ids.shape[0]:
            raise ValueError("boundary faces and boundary ids mismatch after cull")
        
        self.boundary_faces = kept_faces
        return kept_ids
    
    @property
    def n_elements(self) -> int:
        return self.elements.shape[0]

class Mesh():
    """A JAX-based geometric mesh object."""
    # todo: perhaps we want to have this be aa generic class with both 
    # hp-adaptive meshes and static meshes built on top 
    nodes:        Array
    connectivity: ElementConnectivity
    element_type: ElementType
    approx_order: Union[Dict[int, Array], Uninit]

    def __init__(self, mesh: MeshIOMesh, element_type: Union[str, ElementType]):
        if isinstance(element_type, str):
            element_type = ElementType.from_str(element_type)
        
        if not mesh.cells:
            raise ValueError("input mesh has no defined elements")
                
        for i, cell_block in enumerate(mesh.cells):
            if cell_block.type in SUPPORTED_ELEMENTS:
                if element_type.face_type == ElementType.from_str(cell_block.type):
                    faces = jnp.asarray(cell_block.data, dtype = jnp.int32)
                    try:
                        face_entity_ids = mesh.cell_data["CellEntityIds"][i]
                    except:
                        raise MeshReadError("mesh appears to have no boundary faces")

                if element_type == ElementType.from_str(cell_block.type):
                    elements = jnp.asarray(cell_block.data, dtype = jnp.int32)
            
        self.nodes = jnp.asarray(mesh.points, dtype = jnp.float64)
        self.connectivity = ElementConnectivity(elements, faces)
        self.boundary_ids = self.connectivity.cull_non_boundary_faces(face_entity_ids)
        self.element_type = element_type
        self.element_order = Uninit

    def __repr__(self):
        return (
            f"<Mesh object with {self.n_nodes} nodes, {self.n_elements} elements>\n"
            f" - ElementType: {self.element_type}\n"
            f" - Dimensions:  {self.dimensions}\n"
            f" - Coordinate Span: {self.bounds}"
        )

    @classmethod
    def read(cls, filepath: str, element_type: Union[str, ElementType], file_format: Optional[str] = None):
        """Reads a mesh from a file using meshio and returns a Mesh object"""
        meshio_mesh = meshio.read(filepath, file_format)
        return cls(meshio_mesh, element_type)

    def write(self, filepath: str, file_format = None):
        """Writes the mesh to a given filepath using the meshio API"""
        cell_type = None
        for key, value in SUPPORTED_ELEMENTS.items():
            if value == self.element_type:
                cell_type = key
                break 
        
        if cell_type is None:
            raise ValueError(f"could not find a corresponding meshio cell type for {self.element_type}")
        
        points = np.asarray(self.nodes)
        cells = [(cell_type, np.asarray(self.connectivity.elements))]
        
        meshio_mesh = MeshIOMesh(points, cells)
        meshio_mesh.write(filepath, file_format)

    def initialise_element_order(self, order: int):
        """Initialises the dict that stores the local order of each element, {order : elem_ids}"""
        element_ids = jnp.arange(self.n_elements, dtype = jnp.int32)
        self.element_order = {order : element_ids}

    @property
    def n_dofs(self) -> Optional[int]:
        if isinstance(self.element_order, Uninit):
            todo("how do we handle this, either return None or raise an error, idk")
        
        n_dofs = 0
        for order, element_ids in self.element_order.items():
            n_dofs += element_ids.shape[0] * self.element_type.n_dofs(order)
            
        return n_dofs

    @property
    def n_nodes(self) -> int:
        """Returns the number of nodes within the mesh"""
        return self.nodes.shape[0]
    
    @property
    def n_elements(self) -> int:
        """Returns the number of elements within the mesh"""
        return self.connectivity.n_elements
    
    @property
    def dimensions(self) -> int:
        """Returns the dimensions of the mesh"""
        return self.element_type.dimensions
    
    @property
    def bounds(self):
        """Returns the max/min dimensional bounds of the mesh"""
        min_coords = jnp.min(self.nodes, axis = 0)
        max_coords = jnp.max(self.nodes, axis = 0)

        if self.dimensions == 1:
            return (
                (min_coords[0].item(), max_coords[0].item())
            )
        elif self.dimensions == 2:
            return (
                (min_coords[0].item(), max_coords[0].item()),
                (min_coords[1].item(), max_coords[1].item())
            )
        elif self.dimensions == 3:
            return (
                (min_coords[0].item(), max_coords[0].item()),
                (min_coords[1].item(), max_coords[1].item()),
                (min_coords[2].item(), max_coords[2].item())
            )
        else:
            raise ValueError("not sure if this will ever occur")
        