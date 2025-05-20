from mesh import Mesh, simple_1d, simple_2d_rect
from element import Element, ElementType

def main():
    pass

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