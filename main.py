from meshing import Mesh

def main():
    mesh = Mesh.read("structured-double-wedge.vtk", element_type = "quad")
    print(mesh.nodes)
    print(mesh.connectivity)
    print(mesh.element_type)
    mesh.write("wedge-test.vtk")

def _jax_init():
    import jax

    try:
        tpu_devices = jax.devices("tpu")
        print(tpu_devices)
        if tpu_devices:
            jax.config.update("jax_platform_name", "tpu")
            jax.config.update("jax_enable_x64", True)
            return
    except RuntimeError:
        pass

    try:
        gpu_devices = jax.devices("gpu")
        print(gpu_devices)
        if gpu_devices:
            jax.config.update("jax_platform_name", "gpu")
            jax.config.update("jax_enable_x64", True)
            return
    except RuntimeError:
        pass

    jax.config.update("jax_platform_name", "cpu")
    jax.config.update("jax_enable_x64", True)

if __name__ == "__main__":
    _jax_init()
    main()