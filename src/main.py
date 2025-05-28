import os
from meshing import Mesh

def main():
    print(f"JAX_PLATFORMS: {os.environ.get('JAX_PLATFORMS')}")

    mesh = Mesh.read("meshes/structured-double-wedge.vtk", element_type = "quad")
    print(mesh.nodes)
    print(mesh.connectivity)
    print(mesh.element_type)
    mesh.write("wedge-test.vtk")

if __name__ == "__main__":
    main()