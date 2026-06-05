use crate::float::Float;
use ndarray::{Array1, Array2, array};

/// The trait for anything with dimensions
pub trait Dimensioned {
    /// The number of dimensions
    fn dimensions() -> usize;
}

/// The base level trait defining a shape
pub trait Shape<F>: Dimensioned
where
    F: Float,
{
    /// The number of points which define the shape (i.e., the number of corners)
    fn points() -> usize;
    /// The bounds of the reference shape
    fn bounds() -> Array2<F>;
    /// The faces of the shape
    fn faces() -> Vec<Face<F>>;
    /// The extents of the shape
    fn extents() -> (usize, usize) {
        (Self::points(), Self::dimensions())
    }
}

/// A line
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Line {}

/// A triangle
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Tri {}

/// A quadrilateral
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Quad {}

/// A tetrahedron
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Tet {}

/// A hexahedron
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Hex {}

/// A prism
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Pri {}

/// A pyramid
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Pyr {}

/// The definition of a face. The indices are in reference to the shape of which it is a face
#[derive(Debug, Clone)]
pub enum Face<F: Float> {
    Line {
        indices: [usize; 2],
        normal: Array1<F>,
    },
    Tri {
        indices: [usize; 3],
        normal: Array1<F>,
    },
    Quad {
        indices: [usize; 4],
        normal: Array1<F>,
    },
}

macro_rules! dimensioned_impl {
    ($shape:ty, $dimensions:literal) => {
        impl Dimensioned for $shape {
            fn dimensions() -> usize {
                $dimensions
            }
        }
    };
}

dimensioned_impl!(Line, 1);
dimensioned_impl!(Tri, 2);
dimensioned_impl!(Quad, 2);
dimensioned_impl!(Tet, 3);
dimensioned_impl!(Hex, 3);
dimensioned_impl!(Pri, 3);
dimensioned_impl!(Pyr, 3);

impl<F: Float> Shape<F> for Line {
    fn points() -> usize {
        2
    }

    fn bounds() -> Array2<F> {
        array![[-F::one()], [F::one()]]
    }

    fn faces() -> Vec<Face<F>> {
        vec![]
    }
}

impl<F: Float> Shape<F> for Tri {
    fn points() -> usize {
        3
    }

    fn bounds() -> Array2<F> {
        array![
            [-F::one(), -F::one()],
            [F::one(), -F::one()],
            [-F::one(), F::one()],
        ]
    }

    fn faces() -> Vec<Face<F>> {
        vec![
            Face::Line {
                indices: [0, 1],
                normal: array![F::zero(), -F::one()],
            },
            Face::Line {
                indices: [1, 2],
                normal: array![F::one(), -F::one()],
            },
            Face::Line {
                indices: [2, 0],
                normal: array![-F::one(), F::zero()],
            },
        ]
    }
}

impl<F: Float> Shape<F> for Quad {
    fn points() -> usize {
        4
    }

    fn bounds() -> Array2<F> {
        array![
            [-F::one(), -F::one()],
            [F::one(), -F::one()],
            [F::one(), F::one()],
            [-F::one(), F::one()],
        ]
    }

    fn faces() -> Vec<Face<F>> {
        vec![
            Face::Line {
                indices: [0, 1],
                normal: array![F::zero(), -F::one()],
            },
            Face::Line {
                indices: [1, 2],
                normal: array![F::one(), F::zero()],
            },
            Face::Line {
                indices: [2, 3],
                normal: array![F::zero(), F::one()],
            },
            Face::Line {
                indices: [3, 0],
                normal: array![-F::one(), F::zero()],
            },
        ]
    }
}

impl<F: Float> Shape<F> for Tet {
    fn points() -> usize {
        4
    }

    fn bounds() -> Array2<F> {
        array![
            [-F::one(), -F::one(), -F::one()],
            [F::one(), -F::one(), -F::one()],
            [-F::one(), F::one(), -F::one()],
            [-F::one(), -F::one(), F::one()],
        ]
    }

    fn faces() -> Vec<Face<F>> {
        vec![
            Face::Tri {
                indices: [0, 1, 2],
                normal: array![F::zero(), F::zero(), -F::one()],
            },
            Face::Tri {
                indices: [0, 1, 3],
                normal: array![F::zero(), -F::one(), F::zero()],
            },
            Face::Tri {
                indices: [0, 2, 3],
                normal: array![-F::one(), F::zero(), F::zero()],
            },
            Face::Tri {
                indices: [1, 2, 3],
                normal: array![F::one(), F::one(), F::one()],
            },
        ]
    }
}

impl<F: Float> Shape<F> for Hex {
    fn points() -> usize {
        8
    }

    fn bounds() -> Array2<F> {
        array![
            [-F::one(), -F::one(), -F::one()],
            [F::one(), -F::one(), -F::one()],
            [F::one(), F::one(), -F::one()],
            [-F::one(), F::one(), -F::one()],
            [-F::one(), -F::one(), F::one()],
            [F::one(), -F::one(), F::one()],
            [F::one(), F::one(), F::one()],
            [-F::one(), F::one(), F::one()],
        ]
    }

    fn faces() -> Vec<Face<F>> {
        vec![
            Face::Quad {
                indices: [0, 1, 2, 3],
                normal: array![F::zero(), F::zero(), -F::one()],
            },
            Face::Quad {
                indices: [0, 1, 5, 4],
                normal: array![F::zero(), -F::one(), F::zero()],
            },
            Face::Quad {
                indices: [1, 2, 6, 5],
                normal: array![F::one(), F::zero(), F::zero()],
            },
            Face::Quad {
                indices: [2, 3, 7, 6],
                normal: array![F::zero(), F::one(), F::zero()],
            },
            Face::Quad {
                indices: [3, 0, 4, 7],
                normal: array![-F::one(), F::zero(), F::zero()],
            },
            Face::Quad {
                indices: [4, 5, 6, 7],
                normal: array![F::zero(), F::zero(), F::one()],
            },
        ]
    }
}

impl<F: Float> Shape<F> for Pri {
    fn points() -> usize {
        6
    }

    fn bounds() -> Array2<F> {
        array![
            [-F::one(), -F::one(), -F::one()],
            [F::one(), -F::one(), -F::one()],
            [-F::one(), F::one(), -F::one()],
            [-F::one(), -F::one(), F::one()],
            [F::one(), -F::one(), F::one()],
            [-F::one(), F::one(), F::one()]
        ]
    }

    fn faces() -> Vec<Face<F>> {
        vec![
            Face::Tri {
                indices: [0, 1, 2],
                normal: array![F::zero(), F::zero(), -F::one()],
            },
            Face::Tri {
                indices: [3, 4, 5],
                normal: array![F::zero(), F::zero(), F::one()],
            },
            Face::Quad {
                indices: [0, 1, 4, 3],
                normal: array![F::zero(), -F::one(), F::zero()],
            },
            Face::Quad {
                indices: [1, 2, 5, 4],
                normal: array![F::one(), F::one(), F::zero()],
            },
            Face::Quad {
                indices: [2, 0, 3, 5],
                normal: array![-F::one(), F::zero(), F::zero()],
            },
        ]
    }
}

impl<F: Float> Shape<F> for Pyr {
    fn points() -> usize {
        5
    }

    fn bounds() -> Array2<F> {
        array![
            [-F::one(), -F::one(), -F::one()],
            [F::one(), -F::one(), -F::one()],
            [F::one(), F::one(), -F::one()],
            [-F::one(), F::one(), -F::one()],
            [F::zero(), F::zero(), F::one()]
        ]
    }

    fn faces() -> Vec<Face<F>> {
        vec![
            Face::Quad {
                indices: [0, 1, 2, 3],
                normal: array![F::zero(), F::zero(), -F::one()],
            },
            Face::Tri {
                indices: [0, 1, 4],
                normal: array![F::zero(), -F::one(), F::from(0.5).unwrap()],
            },
            Face::Tri {
                indices: [1, 2, 4],
                normal: array![F::one(), F::zero(), F::from(0.5).unwrap()],
            },
            Face::Tri {
                indices: [2, 3, 4],
                normal: array![F::zero(), F::one(), F::from(0.5).unwrap()],
            },
            Face::Tri {
                indices: [3, 0, 4],
                normal: array![-F::one(), F::zero(), F::from(0.5).unwrap()],
            },
        ]
    }
}
