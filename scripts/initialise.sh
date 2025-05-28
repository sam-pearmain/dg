if nvidia-smi -L > /dev/null 2>&1; then 
    export JAX_PLATFORM_NAME="gpu"
else 
    export JAX_PLATFORM_NAME="cpu"
fi

export JAX_ENABLE_X64="true"

echo "JAX_PLATFORM_NAME: $JAX_PLATFORM_NAME"