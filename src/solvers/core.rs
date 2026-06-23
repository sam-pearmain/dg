use std::sync::Arc;

use cudarc::driver::CudaContext;


pub struct CoreSystem {
    device: Arc<CudaContext>, 
}