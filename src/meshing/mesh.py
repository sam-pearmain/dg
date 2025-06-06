import jax.numpy as jnp
import numpy as np
import meshio

from jax import Array
from typing import Optional, Dict, Union
from warnings import warn    
from meshio import Mesh as MeshIOMesh
from utils import todo, isuninit, Uninit, MeshError, MeshReadError
from meshing.boundary import SUPPORTED_BOUNDARY_IDS
from meshing.element import ElementType

class ElementInfo:
    """Element info stored in JAX arrays. Element are each assigned integer values
    to indicate whether they are boundary/interior elements alongside additional info"""
    boundary_info: Array # this cannot be optional as we must define how we treat our boundary elems
    elements_info: Optional[Array]

    def __init__(self, boundary_info: Array, elements_info: Optional[Array] = None):
        if boundary_info is None:
            raise MeshReadError("no specified boundary data, check the mesh")
        self.elements_info = elements_info
        self.boundary_info = boundary_info

    def has_element_info(self) -> bool:
        """Whether the mesh stores internal element info, this is something to be expanded upon"""
        return False if self.elements_info is None else True

    def boundary_info_len(self) -> int:
        """The length of the boundary info array, i.e., how many boundary elements have assigned info"""
        return self.boundary_info.shape[0]

    def elements_info_len(self) -> int:
        """The length of the internal info array"""
        return 0 if self.elements_info is None else self.elements_info.shape[0]


class ElementConnectivity:
    """Element connectivities stored in JAX arrays."""
    elements: Array
    boundary: Array

    def __init__(self, elements: Array, boundary: Array):
        self.elements = elements
        self.boundary = boundary

    def __repr__(self):
        return (
            f"<ElementConnectivity>\n"
            f" - {self.n_elements} elements\n"
            f" - {self.n_boundary_elements} boundary elements"    
        )

    @property
    def n_elements(self) -> int:
        """The number of internal elements"""
        return self.elements.shape[0]
    
    @property
    def n_boundary_elements(self) -> int:
        """The number of boundary elements"""
        return self.boundary.shape[0]

class Mesh():
    """A JAX-based geometric mesh object."""
    # todo: perhaps we want to have this be aa generic class with both 
    # hp-adaptive meshes and static meshes built on top
    nodes:        Array
    connectivity: ElementConnectivity
    element_info: ElementInfo
    element_type: ElementType
    element_order: Union[Dict[int, Array], Uninit]

    def __init__(self, mesh: MeshIOMesh, element_type: Union[str, ElementType]):
        if isinstance(element_type, str):
            element_type = ElementType.from_str(element_type)
        
        self.element_type = element_type

        if not mesh.cells:
            raise ValueError("input mesh has no defined elements")
                
        elems, faces, elem_info, face_info = self._deconstruct_meshio_mesh(mesh)
        connectivity, element_info = self._assemble_elements(elems, faces, elem_info, face_info)
        
        self.nodes = jnp.asarray(mesh.points, dtype = jnp.float64)
        self.connectivity = connectivity
        self.element_info = element_info
        self.element_order = Uninit

        self._sanity_check()

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
        cell_type = self.element_type.as_meshio_str()
        points = np.asarray(self.nodes)
        cells = [(cell_type, np.asarray(self.connectivity.elements))]
        
        meshio_mesh = MeshIOMesh(points, cells)
        meshio_mesh.write(filepath, file_format)

    def _deconstruct_meshio_mesh(self, meshio_mesh: MeshIOMesh) -> tuple[Array, Array, Optional[Array], Array]:
        """Deconstructs the meshio mesh into what we need for our mesh representation"""
        for i, cell_block in enumerate(meshio_mesh.cells):
            if cell_block.type == self.element_type.as_meshio_str():
                elements = jnp.asarray(cell_block.data, dtype = jnp.int32)
                elems_idx = i

            if cell_block.type == self.element_type.face_type.as_meshio_str():
                faces = jnp.asarray(cell_block.data, dtype = jnp.int32)
                faces_idx = i

        try:
            raw_elements_info = meshio_mesh.cell_data["CellEntityIds"][elems_idx]
            raw_face_info = meshio_mesh.cell_data["CellEntityIds"][faces_idx]
        except:
            raise MeshReadError("mesh appears to have no boundary data, update mesh and tag boundaries")

        elements_info = jnp.asarray(raw_elements_info, dtype = jnp.int32).reshape(-1)
        face_info = jnp.asarray(raw_face_info, dtype = jnp.int32).reshape(-1)

        if bool(jnp.all(elements_info == -1)):
            elements_info = None
        if bool(jnp.all(face_info == -1)):
            raise MeshReadError("boundary faces are unassigned, unknown boundary, check mesh")

        return elements, faces, elements_info, face_info

    def _assemble_elements(
            self, elements: Array, faces: Array, 
            elem_info: Optional[Array], face_info: Array
        ) -> tuple[ElementConnectivity, ElementInfo]:
        """Assembles the connectivity and element info classes, deleting faces 
        which do not comprise of the boundary in the process"""
        allowed_tags = jnp.asarray(list(SUPPORTED_BOUNDARY_IDS.keys()), dtype = jnp.int32)
        mask = (face_info != -1) & jnp.isin(face_info, allowed_tags)
        
        # extract the boundary
        boundary = faces[mask]
        boundary_info = face_info[mask]

        invalid_tags = sorted(
            set(face_info.tolist()) - 
            set(allowed_tags.tolist()) - 
            {-1}
        )
        if invalid_tags:
            warn(f"ignoring unsupported boundary id/s {invalid_tags}")

        connectivity = ElementConnectivity(elements, boundary)
        element_info = ElementInfo(boundary_info, elem_info)

        return connectivity, element_info

    def _initialise_element_order(self, order: int):
        """Initialises the dict that stores the local order of each element, {order : elem_ids}"""
        element_ids = jnp.arange(self.n_elements, dtype = jnp.int32)
        self.element_order = {order : element_ids}

    def _sanity_check(self):
        """Perform a sanity check on the mesh"""
        min_ref = self.connectivity.elements.min()
        max_ref = self.connectivity.elements.max()
        if min_ref < 0 or max_ref >= self.n_nodes:
            raise MeshError(
                f"element connectivity index out of range: "
                f"found refs [{min_ref},{max_ref}], valid [0,{self.n_nodes - 1}]"
            )
        
        n_boundary_faces = self.connectivity.n_boundary_elements
        n_boundary_tags = self.element_info.boundary_info_len()
        if n_boundary_faces != n_boundary_tags:
            raise MeshError(
                f"boundary faces ({n_boundary_faces}) and boundary tags ({n_boundary_tags}) mismatch"
            )

    @property
    def n_dofs(self) -> Optional[int]:
        """Return the number of degrees of freedom in the mesh"""
        if isuninit(self.element_order):
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
    def boundary_element_type(self) -> ElementType:
        """Returns the type of boundary elements"""
        return self.element_type.face_type

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
        