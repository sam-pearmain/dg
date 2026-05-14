use std::fs;

use dg::gmsh::{mshfile::Msh, parsers::MshParser};


fn main() {
    let bytes = fs::read("meshes/cube-ascii.msh").expect("what");
    let msh = Msh::<usize, i32, f64>::try_from(bytes.as_slice()).expect(":(");
}
