import jax
from meshing import Mesh

def main():
    print(jax.config.read)

    mesh = Mesh.read("meshes/structured-double-wedge.vtk", element_type = "quad")
    print(mesh.nodes)
    print(mesh.connectivity)
    print(mesh.element_type)
    mesh.write("wedge-test.vtk")

if __name__ == "__main__":
    main()