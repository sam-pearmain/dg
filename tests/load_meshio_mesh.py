from meshio import read

def test():
    mesh = read("meshes/box-mesh.vtk")
    print(mesh.cells)
    print(mesh.cell_data.data)
    # for i, (cell, data) in enumerate(mesh.cell_data.items()):
    #     print(i, cell, data)
    #     print(data[1])

if __name__ == "__main__":
    test()