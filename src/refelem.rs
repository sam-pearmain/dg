use crate::shapes::{Dimensioned, Shape};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReferenceElement {
    shape: Shape,
    order: usize,
}

impl Dimensioned for ReferenceElement {
    fn dimensions(&self) -> usize {
        self.shape.dimensions()
    }
}

impl ReferenceElement {
    fn new(shape: Shape, order: usize) -> Self {
        Self { shape, order }
    }

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