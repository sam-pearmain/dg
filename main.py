from mesh import Mesh, _simple_1d
from element import Element, ElementType

def main():
    mesh = _simple_1d(2, 101)
    print(mesh.elements)
    mesh.plot()

if __name__ == "__main__":
    main()