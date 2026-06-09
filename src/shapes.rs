use std::marker::PhantomData;

use crate::float::Float;
use ndarray::{Array1, Array2, array};

/// The trait for anything with dimensions
pub trait Dimensioned {
    /// The number of dimensions
    fn ndims() -> usize;
}

/// The base level trait defining a shape
pub trait Shape<F>: Dimensioned
where
    F: Float,
{
    /// The number of points which define the shape (i.e., the number of corners)
    fn npoints() -> usize;
    /// The number of basis functions (i.e., the number of solution points) for a given order
    fn nbases_from_order(order: usize) -> usize;
    /// The bounds of the reference shape
    fn bounds() -> Array2<F>;
    /// The faces of the shape
    fn faces() -> Vec<Face<F>>;
    /// The extents of the shape
    fn extents() -> (usize, usize) {
        (Self::npoints(), Self::ndims())
    }
}

/// An enum for the shapes
pub enum ShapeKind<F: Float> {
    Line(Line<F>),
    Tri(Tri<F>),
    Quad(Quad<F>),
    Tet(Tet<F>),
    Hex(Hex<F>),
    Pri(Pri<F>),
    Pyr(Pyr<F>),
}

/// A line
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Line<F: Float> {
    _marker: PhantomData<F>,
}

/// A triangle
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Tri<F: Float> {
    _marker: PhantomData<F>,
}

/// A quadrilateral
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Quad<F: Float> {
    _marker: PhantomData<F>,
}

/// A tetrahedron
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Tet<F: Float> {
    _marker: PhantomData<F>,
}

/// A hexahedron
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Hex<F: Float> {
    _marker: PhantomData<F>,
}

/// A prism
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Pri<F: Float> {
    _marker: PhantomData<F>,
}

/// A pyramid
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Pyr<F: Float> {
    _marker: PhantomData<F>,
}

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

impl<F: Float> Face<F> {
    fn n_points_from_order(&self, order: usize) -> usize {
        match &self {
            Self::Line {
                indices: _,
                normal: _,
            } => order + 1,
            Self::Tri {
                indices: _,
                normal: _,
            } => (order + 1) * (order + 2) / 2,
            Self::Quad {
                indices: _,
                normal: _,
            } => (order + 1).pow(2),
        }
    }
}

impl<F: Float> Dimensioned for Line<F> {
    fn ndims() -> usize {
        1
    }
}

impl<F: Float> Dimensioned for Tri<F> {
    fn ndims() -> usize {
        2
    }
}

impl<F: Float> Dimensioned for Quad<F> {
    fn ndims() -> usize {
        2
    }
}

impl<F: Float> Dimensioned for Tet<F> {
    fn ndims() -> usize {
        3
    }
}

impl<F: Float> Dimensioned for Hex<F> {
    fn ndims() -> usize {
        3
    }
}

impl<F: Float> Dimensioned for Pri<F> {
    fn ndims() -> usize {
        3
    }
}

impl<F: Float> Dimensioned for Pyr<F> {
    fn ndims() -> usize {
        3
    }
}

impl<F: Float> Shape<F> for Line<F> {
    fn npoints() -> usize {
        2
    }

    fn nbases_from_order(order: usize) -> usize {
        order + 1
    }

    fn bounds() -> Array2<F> {
        array![[-F::one()], [F::one()]]
    }

    fn faces() -> Vec<Face<F>> {
        vec![]
    }
}

impl<F: Float> Shape<F> for Tri<F> {
    fn npoints() -> usize {
        3
    }

    fn nbases_from_order(order: usize) -> usize {
        (order + 1) * (order + 2) / 2
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

impl<F: Float> Shape<F> for Quad<F> {
    fn npoints() -> usize {
        4
    }

    fn nbases_from_order(order: usize) -> usize {
        (order + 1).pow(2)
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

impl<F: Float> Shape<F> for Tet<F> {
    fn npoints() -> usize {
        4
    }

    fn nbases_from_order(order: usize) -> usize {
        (order + 1) * (order + 2) * (order + 3) / 6
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

impl<F: Float> Shape<F> for Hex<F> {
    fn npoints() -> usize {
        8
    }

    fn nbases_from_order(order: usize) -> usize {
        (order + 1).pow(3)
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

impl<F: Float> Shape<F> for Pri<F> {
    fn npoints() -> usize {
        6
    }

    fn nbases_from_order(order: usize) -> usize {
        (order + 1) * (order + 1) * (order + 2) / 2
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

impl<F: Float> Shape<F> for Pyr<F> {
    fn npoints() -> usize {
        5
    }

    fn nbases_from_order(order: usize) -> usize {
        (order + 1) * (order + 2) * (2 * order + 3) / 6
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
