pub enum BaseShape {
    Line,
    Triangle,
    Quadrilateral,
    Tetrahedron,
    Hexahedron,
    Prism,
    Pyramid,
}

impl BaseShape {
    pub fn ndims(&self) -> usize {
        match self {
            Self::Line => 1,
            Self::Triangle => 2,
            Self::Quadrilateral => 2,
            _ => 3,
        }
    }
}

pub trait Shape {
    const BASE_SHAPE: BaseShape;

    fn n_solution_points_from_order(order: usize) -> usize;
    fn order_from_n_solution_points(n_pts: usize) -> usize;
}

// impl Shape for Line2 {}

// impl Shape for Line3 {}
