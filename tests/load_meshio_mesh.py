from meshio import read

def test():
    mesh = read("meshes/structured-double-wedge.vtk")
    print(mesh.cells)
    print(mesh.cell_data)
    

if __name__ == "__main__":
    test()