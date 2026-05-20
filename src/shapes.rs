use std::str::FromStr;

use anyhow::Error;
use ndarray::{Array2, ArrayView1};
use num::Float;

use crate::polys::Basis;

/// The seven basic shapes
pub enum Shape {
    Line,
    Triangle,
    Quadrilateral,
    Tetrahedron,
    Hexahedron,
    Prism,
    Pyramid,
}

impl Shape {
    pub fn dimensions(&self) -> usize {
        match self {
            Self::Line => 1,
            Self::Triangle | Self::Quadrilateral => 2,
            _ => 3,
        }
    }
}

impl<F: Float> Basis<F> for Shape {
    fn orthonormal_basis_at(points: ArrayView1<'_, F>) -> Array2<F> {
        todo!()
    }
}

impl FromStr for Shape {
    type Err = Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_lowercase() {
            "line" => Self::Line
        }
    }
}