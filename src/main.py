from meshing import Mesh

def main():
    mesh = Mesh.read("meshes/structured-double-wedge.vtk", element_type = "quad")
    print(mesh)
    print(mesh.boundary_ids)
    mesh.write("wedge-test.vtk")

if __name__ == "__main__":
    main()