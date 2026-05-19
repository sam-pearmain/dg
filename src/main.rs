use std::{fs, time::Instant};

use anyhow::{Ok, Result};
use dg::gmsh::mshfile::Msh;

fn main() -> Result<()> {
    let start = Instant::now();
    let bytes = fs::read("meshes/cube-ascii.msh")?;
    let _msh = Msh::<u64, i64, f64>::try_from(bytes.as_slice())?;

    println!("elapsed: {:?}", start.elapsed());
    Ok(())
}
