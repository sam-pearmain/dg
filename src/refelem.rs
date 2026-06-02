use std::marker::PhantomData;

use num::Float;

use crate::shapes::{Dimensioned, Shapes};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ReferenceElement<F: Float> {
    shape: Shapes,
    order: usize,
    _marker: PhantomData<F>,
}

impl<F: Float> Dimensioned for ReferenceElement<F> {
    fn dimensions(&self) -> usize {
        self.shape.dimensions()
    }
}

impl<F: Float> ReferenceElement<F> {
    fn new(shape: Shape, order: usize) -> Self {
        Self {
            shape,
            order,
            _marker: PhantomData,
        }
    }

    /// The number of solution points
    fn n_solution_points(&self) -> usize {
        match self.shape {
            Shape::Line => self.order + 1,
            Shape::Triangle => (self.order + 1) * (self.order + 2) / 2,
            Shape::Quadrilateral => (self.order + 1) * (self.order + 1),
            Shape::Tetrahedron => (self.order + 1) * (self.order + 2) * (self.order + 3) / 6,
            Shape::Hexahedron => (self.order + 1) * (self.order + 1) * (self.order + 1),
            Shape::Prism => (self.order + 1) * (self.order + 1) * (self.order + 2) / 2,
            Shape::Pyramid => (self.order + 1) * (self.order + 2) * (2 * self.order + 3) / 6,
        }
    }
}
