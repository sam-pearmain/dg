from mesh import Mesh, simple_1d, simple_2d_rect
from element import Element, ElementType

def main():
    mesh = simple_1d(2, 101)
    mesh_2d = simple_2d_rect(1, 1, 101, 101)

    print(mesh.connectivity)
    print(mesh.n_elements())

    print(mesh_2d.connectivity)
    print(mesh_2d.n_elements())

def _jax_init():
    import jax

    jax.config.update("jax_enable_x64", True)

    try:
        tpu_devices = jax.devices("tpu")
        if tpu_devices:
            jax.config.update("jax_platform_name", "tpu")
            print("using tpu jax backend")
            return
    except RuntimeError:
        pass

    try:
        gpu_devices = jax.devices("gpu")
        if gpu_devices:
            jax.config.update("jax_platform_name", "gpu")
            print("using gpu jax backend")
            return
    except RuntimeError:
        pass

    jax.config.update("jax_platform_name", "cpu")
    print("using cpu jax backend")

if __name__ == "__main__":
    _jax_init()
    main()