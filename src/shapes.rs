use ndarray::{Array2, ArrayView1};
use num::Float;

use crate::polys::Basis;

/// The seven reference shapes
pub enum Shape {
    Line { order: usize },
    Triangle { order: usize },
    Quadrilateral { order: usize },
    Tetrahedron { order: usize },
    Hexahedron { order: usize },
    Prism { order: usize },
    Pyramid { order: usize },
}

impl Shape {
    pub fn dimensions(&self) -> usize {
        match self {
            Self::Line { order: _ } => 1,
            Self::Triangle { order: _ } | Self::Quadrilateral { order: _ } => 2,
            _ => 3,
        }
    }
}

impl<F: Float> Basis<F> for Shape {
    fn orthonormal_basis_at(points: ArrayView1<'_, F>) -> Array2<F> {
        todo!()
    }
}
