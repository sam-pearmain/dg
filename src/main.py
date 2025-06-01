from meshing import Mesh

def main():
    mesh = Mesh.read("meshes/box-mesh.vtk", element_type = "quad")
    print(mesh)
    mesh.write("wedge-test.vtk")

if __name__ == "__main__":
    main()