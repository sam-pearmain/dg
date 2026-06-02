use std::marker::PhantomData;

use num::Float;

use crate::shapes::{Dimensioned, Line, ReferenceShape, Shape};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ReferenceElement<F: Float, S: ReferenceShape<F> + Dimensioned> {
    shape: S, 
    order: usize,
    _marker: PhantomData<(F, S)>
}

impl<F: Float, S: ReferenceShape<F> + Dimensioned> Dimensioned for ReferenceElement<F, S> {
    fn dimensions(&self) -> usize {
        self.shape.dimensions()
    }
}

impl<F: Float, S: ReferenceShape<F> + Dimensioned> ReferenceElement<F, S> {
    fn new(order: usize) -> Self {
        Self { order }
    }

    fn n_solution_points(&self) -> usize {
        match self.shape {
            S::Line => self.order + 1, 
            Shape::Triangle => (self.order + 1) * (self.order + 2) / 2, 
            Shape::Quadrilateral => (self.order + 1) * (self.order + 1), 
            Shape::Tetrahedron => (self.order + 1) * (self.order + 2) * (self.order + 3) / 6, 
            Shape::Hexahedron => (self.order + 1) * (self.order + 1) * (self.order + 1), 
            Shape::Prism => (self.order + 1) * (self.order + 1) * (self.order + 2) / 2, 
            Shape::Pyramid => (self.order + 1) * (self.order + 2) * (2 * self.order + 3) / 6,
        }
    }
}

pub type ReferenceLine<F: Float> = ReferenceElement<F, Line<F>>;