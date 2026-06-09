use std::marker::PhantomData;

use ndarray::Array2;

use crate::float::Float;
use crate::operators::Operators;
use crate::shapes::{Hex, Line, Pyr, Quad, Shape, Tet, Tri};

/// The reference element trait
pub trait ReferenceElement<F: Float>: Operators<F> {
    type Shape: Shape<F>;

    /// The solution points
    fn solution_points(&self) -> Array2<F>;
    /// The flux points
    fn flux_points(&self) -> Array2<F>;
    /// The quadrature points
    fn quadrature_points(&self) -> Array2<F>;
}

pub struct ReferenceShape<F: Float, S: Shape<F>> {
    pub order: usize,
    _marker: PhantomData<(F, S)>,
}

/// A reference line
#[allow(type_alias_bounds)]
pub type ReferenceLine<F: Float> = ReferenceShape<F, Line<F>>;
/// A reference triangle
#[allow(type_alias_bounds)]
pub type ReferenceTri<F: Float> = ReferenceShape<F, Tri<F>>;
/// A reference quadrilateral
#[allow(type_alias_bounds)]
pub type ReferenceQuad<F: Float> = ReferenceShape<F, Quad<F>>;
/// A reference tetrahedron
#[allow(type_alias_bounds)]
pub type ReferenceTet<F: Float> = ReferenceShape<F, Tet<F>>;
/// A reference hexahedron
#[allow(type_alias_bounds)]
pub type ReferenceHex<F: Float> = ReferenceShape<F, Hex<F>>;
/// A reference prism
#[allow(type_alias_bounds)]
pub type ReferencePri<F: Float> = ReferenceShape<F, Tet<F>>;
/// A reference pyramid
#[allow(type_alias_bounds)]
pub type ReferencePyr<F: Float> = ReferenceShape<F, Pyr<F>>;

impl<F: Float, S: Shape<F>> ReferenceShape<F, S> {
    fn new(order: usize) -> Self {
        Self {
            order,
            _marker: PhantomData,
        }
    }
}
