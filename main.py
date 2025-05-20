import meshio
from mesh import Mesh
from element import Element, ElementType

def main():
    _jax_init()

    points = [
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
    ]
    cells = [
        ("triangle", [[0, 1, 2], [1, 3, 2]])
    ]

    mesh = meshio.Mesh(points, cells)
    jax_mesh = Mesh(mesh)

    print(jax_mesh.dimensions, jax_mesh.element_type)
    print(jax_mesh.nodes)
    print(jax_mesh.connectivity)

    jax_mesh.write("jax_mesh.vtk")

def _jax_init():
    import jax

    jax.config.update("jax_enable_x64", True)

    try:
        tpu_devices = jax.devices("tpu")
        if tpu_devices:
            jax.config.update("jax_platform_name", "tpu")
            return
    except RuntimeError:
        pass

    try:
        gpu_devices = jax.devices("gpu")
        if gpu_devices:
            jax.config.update("jax_platform_name", "gpu")
            return
    except RuntimeError:
        pass

    jax.config.update("jax_platform_name", "cpu")

if __name__ == "__main__":
    _jax_init()
    main()