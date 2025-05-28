if nvidia-smi -L > /dev/null 2>&1; then
  export JAX_PLATFORMS="gpu"
else
  export JAX_PLATFORMS="cpu"
fi

export JAX_ENABLE_X64="true"