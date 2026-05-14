use std::{fs, time::Instant};

use anyhow::{Ok, Result};
use dg::gmsh::mshfile::Msh;

fn main() -> Result<()> {
    let start = Instant::now();
    let bytes = fs::read("meshes/cube-binary.msh")?;
    let msh = Msh::<u32, i32, f32>::try_from(bytes.as_slice())?;
    
    println!("elapsed: {:?}", start.elapsed());
    Ok(())
}
