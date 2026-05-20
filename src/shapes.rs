use crate::polys::Basis;
use ndarray::{Array1, Array2, ArrayView1};
use num::Float;

pub trait Dimensioned {
    fn dimensions(&self) -> usize;
}

/// The seven basic shapes
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Shape {
    Line,
    Triangle,
    Quadrilateral,
    Tetrahedron,
    Hexahedron,
    Prism,
    Pyramid,
}

/// The definition of a face. The indices are in reference to the shape of which it is a face
#[derive(Debug, Clone)]
pub enum Face<F: Float> {
    Line {
        indices: [usize; 2],
        normal: Array1<F>,
    },
    Triangle {
        indices: [usize; 3],
        normal: Array1<F>,
    },
    Quadrilateral {
        indices: [usize; 4],
        normal: Array1<F>,
    },
}

impl Dimensioned for Shape {
    fn dimensions(&self) -> usize {
        match self {
            Self::Line => 1,
            Self::Triangle | Self::Quadrilateral => 2,
            _ => 3,
        }
    }
}

impl<F: Float> Dimensioned for Face<F> {
    fn dimensions(&self) -> usize {
        match self {
            Self::Line { .. } => 1,
            _ => 2,
        }
    }
}

impl Shape {
    /// The number of points which define the shape (i.e., the number of corners)
    pub fn points(&self) -> usize {
        match self {
            Self::Line => 2,
            Self::Triangle => 3,
            Self::Quadrilateral => 4,
            Self::Tetrahedron => 4,
            Self::Hexahedron => 8,
            Self::Prism => 6,
            Self::Pyramid => 5,
        }
    }

    /// The bounds of the reference shape
    #[rustfmt::skip]
    pub fn bounds<'a, F: Float>(&self) -> Array2<F> {
        let zero = F::zero();
        let one = F::one();

        let extents = (self.points(), self.dimensions());

        match self {
            Self::Line => Array2::from_shape_vec(extents, vec![
                -one, 
                one
            ]).unwrap(), 
            Self::Triangle => Array2::from_shape_vec(extents, vec![
                -one, -one, 
                one, -one, 
                -one, one
            ]).unwrap(),
            Self::Quadrilateral => Array2::from_shape_vec(extents, vec![
                -one, -one, 
                one, -one, 
                one, one, 
                -one, one
            ]).unwrap(), 
            Self::Tetrahedron => Array2::from_shape_vec(extents, vec![
                -one, -one, -one, 
                one, -one, -one, 
                -one, one, -one, 
                -one, -one, one, 
            ]).unwrap(), 
            Self::Hexahedron => Array2::from_shape_vec(extents, vec![
                -one, -one, -one, 
                one, -one, -one, 
                one, one, -one, 
                -one, one, -one, 
                -one, -one, one, 
                one, -one, one, 
                one, one, one, 
                -one, one, one
            ]).unwrap(),
            Self::Prism => Array2::from_shape_vec(extents, vec![
                -one, -one, -one, 
                one, -one, -one, 
                -one, one, -one, 
                -one, -one, one, 
                one, -one, one, 
                -one, one, one
            ]).unwrap(),
            Self::Pyramid => Array2::from_shape_vec(extents, vec![
                -one, -one, -one, 
                one, -one, -one, 
                one, one, -one, 
                -one, one, -one, 
                zero, zero, one
            ]).unwrap(),
        }
    }

    /// The faces of the shape
    pub fn faces<F: Float>(&self) -> Vec<Face<F>> {
        let zero = F::zero();
        let one = F::one();
        let half = F::from(0.5).unwrap();

        match self {
            Self::Line => vec![],
            Self::Triangle => vec![
                Face::Line {
                    indices: [0, 1],
                    normal: Array1::from_vec(vec![zero, -one]),
                },
                Face::Line {
                    indices: [1, 2],
                    normal: Array1::from_vec(vec![one, one]),
                },
                Face::Line {
                    indices: [2, 0],
                    normal: Array1::from_vec(vec![-one, zero]),
                },
            ],
            Self::Quadrilateral => vec![
                Face::Line {
                    indices: [0, 1],
                    normal: Array1::from_vec(vec![zero, -one]),
                },
                Face::Line {
                    indices: [1, 2],
                    normal: Array1::from_vec(vec![one, zero]),
                },
                Face::Line {
                    indices: [2, 3],
                    normal: Array1::from_vec(vec![zero, one]),
                },
                Face::Line {
                    indices: [3, 0],
                    normal: Array1::from_vec(vec![-one, zero]),
                },
            ],
            Self::Tetrahedron => vec![
                Face::Triangle {
                    indices: [0, 1, 2],
                    normal: Array1::from_vec(vec![zero, zero, -one]),
                },
                Face::Triangle {
                    indices: [0, 1, 3],
                    normal: Array1::from_vec(vec![zero, -one, zero]),
                },
                Face::Triangle {
                    indices: [0, 2, 3],
                    normal: Array1::from_vec(vec![-one, zero, zero]),
                },
                Face::Triangle {
                    indices: [1, 2, 3],
                    normal: Array1::from_vec(vec![one, one, one]),
                },
            ],
            Self::Hexahedron => vec![
                Face::Quadrilateral {
                    indices: [0, 1, 2, 3],
                    normal: Array1::from_vec(vec![zero, zero, -one]),
                },
                Face::Quadrilateral {
                    indices: [0, 1, 5, 4],
                    normal: Array1::from_vec(vec![zero, -one, zero]),
                },
                Face::Quadrilateral {
                    indices: [1, 2, 6, 5],
                    normal: Array1::from_vec(vec![one, zero, zero]),
                },
                Face::Quadrilateral {
                    indices: [2, 3, 7, 6],
                    normal: Array1::from_vec(vec![zero, one, zero]),
                },
                Face::Quadrilateral {
                    indices: [3, 0, 4, 7],
                    normal: Array1::from_vec(vec![-one, zero, zero]),
                },
                Face::Quadrilateral {
                    indices: [4, 5, 6, 7],
                    normal: Array1::from_vec(vec![zero, zero, one]),
                },
            ],
            Self::Prism => vec![
                Face::Triangle {
                    indices: [0, 1, 2],
                    normal: Array1::from_vec(vec![zero, zero, -one]),
                },
                Face::Triangle {
                    indices: [3, 4, 5],
                    normal: Array1::from_vec(vec![zero, zero, one]),
                },
                Face::Quadrilateral {
                    indices: [0, 1, 4, 3],
                    normal: Array1::from_vec(vec![zero, -one, zero]),
                },
                Face::Quadrilateral {
                    indices: [1, 2, 5, 4],
                    normal: Array1::from_vec(vec![one, one, zero]),
                },
                Face::Quadrilateral {
                    indices: [2, 0, 3, 5],
                    normal: Array1::from_vec(vec![-one, zero, zero]),
                },
            ],
            Self::Pyramid => vec![
                Face::Quadrilateral {
                    indices: [0, 1, 2, 3],
                    normal: Array1::from_vec(vec![zero, zero, -one]),
                },
                Face::Triangle {
                    indices: [0, 1, 4],
                    normal: Array1::from_vec(vec![zero, -one, half]),
                },
                Face::Triangle {
                    indices: [1, 2, 4],
                    normal: Array1::from_vec(vec![one, zero, half]),
                },
                Face::Triangle {
                    indices: [2, 3, 4],
                    normal: Array1::from_vec(vec![zero, one, half]),
                },
                Face::Triangle {
                    indices: [3, 0, 4],
                    normal: Array1::from_vec(vec![-one, zero, half]),
                },
            ],
        }
    }
}

impl<F: Float> Basis<F> for Shape {
    fn orthonormal_basis_at(points: ArrayView1<'_, F>) -> Array2<F> {
        todo!()
    }
}
