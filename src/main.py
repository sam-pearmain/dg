from meshing import Mesh

def main():
    mesh = Mesh.read("structured-double-wedge.vtk", element_type = "quad")
    print(mesh.nodes)
    print(mesh.connectivity)
    print(mesh.element_type)
    mesh.write("wedge-test.vtk")

if __name__ == "__main__":
    main()