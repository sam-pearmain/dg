from mesh import Mesh, simple_1d, simple_2d_rect
from element import Element, ElementType

def main():
    mesh = simple_1d(2, 101)
    mesh_2d = simple_2d_rect(1, 1, 10001, 10001)

    print(mesh.connectivity)
    print(mesh.n_elements())

    print(mesh_2d.connectivity)
    print(mesh_2d.n_elements())

if __name__ == "__main__":
    main()