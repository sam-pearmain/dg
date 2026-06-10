use cudarc::{
    driver::{CudaContext, LaunchConfig, PushKernelArg},
    nvrtc::compile_ptx,
};
use std::sync::Arc;
use std::time::Instant;

fn profile_block_size(ctx: &Arc<CudaContext>, block_size: u32, numel: usize) -> f64 {
    let kernel_template = r#"
        extern "C" __global__ void compute_kernel(float* out, const float* in, int n) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < n) {
                out[idx] = in[idx] * 2.0f;
            }
        }
    "#;

    let full_source = format!(
        "#ifndef BLOCK_SIZE\n#define BLOCK_SIZE {}\n#endif\n{}",
        block_size, kernel_template
    );

    // ctx is now &Arc<CudaContext>, so these methods will resolve correctly
    let ptx = compile_ptx(full_source).unwrap();
    let module = ctx.load_module(ptx).unwrap();
    let stream = ctx.default_stream();

    let host_in = vec![1.0f32; numel];
    let host_out = vec![0.0f32; numel];

    let dev_in = stream.clone_htod(&host_in).unwrap();
    let mut dev_out = stream.clone_htod(&host_out).unwrap();

    let f = module.load_function("compute_kernel").unwrap();

    let blocks_per_grid = (numel as u32 + block_size - 1) / block_size;
    let cfg = LaunchConfig {
        grid_dim: (blocks_per_grid, 1, 1),
        block_dim: (block_size, 1, 1),
        shared_mem_bytes: 0,
    };

    let numel_i32 = numel as i32;

    // Warm-up
    for _ in 0..10 {
        let mut launch_args = stream.launch_builder(&f);
        launch_args.arg(&mut dev_out);
        launch_args.arg(&dev_in);
        launch_args.arg(&numel_i32);
        unsafe { launch_args.launch(cfg) }.unwrap();
    }
    stream.synchronize().unwrap();

    // Profile
    let iterations = 100;
    let start_time = Instant::now();

    for _ in 0..iterations {
        let mut launch_args = stream.launch_builder(&f);
        launch_args.arg(&mut dev_out);
        launch_args.arg(&dev_in);
        launch_args.arg(&numel_i32);
        unsafe { launch_args.launch(cfg) }.unwrap();
    }
    stream.synchronize().unwrap();

    let total_duration = start_time.elapsed();
    total_duration.as_secs_f64() / (iterations as f64)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ctx = CudaContext::new(0)?; // ctx is an Arc<CudaContext>
    let elements_count = 1_000_000;
    let candidates = vec![32, 64, 128, 256, 512, 1024];

    let mut optimal_size = 32;
    let mut lowest_latency = f64::MAX;

    for size in candidates {
        let avg_time = profile_block_size(&ctx, size, elements_count);
        println!(
            "Block Size: {:4} | Execution Time: {:.6} ms",
            size,
            avg_time * 1000.0
        );

        if avg_time < lowest_latency {
            lowest_latency = avg_time;
            optimal_size = size;
        }
    }

    println!("Optimal block configuration selected: {}", optimal_size);
    Ok(())
}
