from meshing import Mesh
from utils import _jax_init
import numerics.quadrature.rules

def main():
    mesh = Mesh.read("structured-double-wedge.vtk", element_type = "quad")
    print(mesh.nodes)
    print(mesh.connectivity)
    print(mesh.element_type)
    mesh.write("wedge-test.vtk")

if __name__ == "__main__":
    _jax_init()
    main()