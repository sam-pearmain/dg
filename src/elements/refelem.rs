use std::marker::PhantomData;
use std::sync::OnceLock;

use ndarray::{Array2, ArrayView2};

use crate::elements::basis::{Basis, LineBasis, QuadBasis};
use crate::float::Float;
use crate::elements::operators::Operators;
use crate::elements::shapes::{Hex, Line, Pyr, Quad, Shape, Tet, Tri};

/// The reference element
pub trait ReferenceElement<F: Float>: Operators<F> {
    type Shape: Shape<F>;

    /// The solution points
    fn solution_points(&self) -> ArrayView2<'_, F>;
    /// The flux points
    fn flux_points(&self) -> ArrayView2<'_, F>;
    /// The quadrature points
    fn quadrature_points(&self) -> ArrayView2<'_, F>;
}

pub struct ReferenceShape<F: Float, S: Shape<F>> {
    pub order: usize,
    solution_points: OnceLock<Array2<F>>,
    flux_points: OnceLock<Array2<F>>,
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
