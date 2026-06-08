use std::marker::PhantomData;

use ndarray::{Array2, ArrayView2, Axis};
use ndarray_linalg::Inverse;

use crate::float::Float;
use crate::polys::legendre;
use crate::shapes::{Hex, Line, Pri, Pyr, Quad, Shape, Tet, Tri};

pub trait Basis<F>
where
    F: Float,
{
    /// The orthonormal basis at given points
    fn orthonormal_basis(order: usize, points: ArrayView2<'_, F>) -> Array2<F>;
    /// The derivative of the orthonormal basis at given points
    fn grad_orthonormal_basis(order: usize, points: ArrayView2<'_, F>) -> Array2<F>;
    /// The Vandermonde matrix at given points
    fn vandermonde(order: usize, nodals: ArrayView2<'_, F>) -> Array2<F> {
        Self::orthonormal_basis(order, nodals)
    }
    /// The inverted Vandermonde matrix at given points
    fn inverted_vandermonde(order: usize, nodals: ArrayView2<'_, F>) -> Array2<F> {
        Self::vandermonde(order, nodals)
            .inv()
            .expect("failed vandermonde inversion, check precision and polynomial order")
    }
    /// The nodal basis at given points
    fn nodal_basis(
        order: usize,
        points: ArrayView2<'_, F>,
        nodals: ArrayView2<'_, F>,
    ) -> Array2<F> {
        Self::orthonormal_basis(order, points).dot(&Self::inverted_vandermonde(order, nodals))
    }
    /// The derivative of the nodal basis at given points
    fn grad_nodal_basis(
        order: usize,
        points: ArrayView2<'_, F>,
        nodals: ArrayView2<'_, F>,
    ) -> Array2<F> {
        Self::grad_orthonormal_basis(order, points).dot(&Self::inverted_vandermonde(order, nodals))
    }
}

///
pub struct ShapeBasis<F: Float, S: Shape<F>> {
    _marker: PhantomData<(F, S)>,
}

/// todo
pub struct BasisCache<F: Float> {
    _marker: PhantomData<F>,
}

/// todo
pub struct CachedShapeBasis<F: Float, S: Shape<F>> {
    cache: BasisCache<F>,
    _marker: PhantomData<(F, S)>,
}

/// The line basis
#[allow(type_alias_bounds)]
pub type LineBasis<F: Float> = ShapeBasis<F, Line<F>>;

/// The triangle basis
#[allow(type_alias_bounds)]
pub type TriBasis<F: Float> = ShapeBasis<F, Tri<F>>;

/// The quadrilateral basis
#[allow(type_alias_bounds)]
pub type QuadBasis<F: Float> = ShapeBasis<F, Quad<F>>;

/// The tetrahedral basis
#[allow(type_alias_bounds)]
pub type TetBasis<F: Float> = ShapeBasis<F, Tet<F>>;

/// The hexahedral basis
#[allow(type_alias_bounds)]
pub type HexBasis<F: Float> = ShapeBasis<F, Hex<F>>;

/// The prismatic basis
#[allow(type_alias_bounds)]
pub type PriBasis<F: Float> = ShapeBasis<F, Pri<F>>;

/// The pyramid basis
#[allow(type_alias_bounds)]
pub type PyrBasis<F: Float> = ShapeBasis<F, Pyr<F>>;

impl<F: Float> Basis<F> for LineBasis<F> {
    fn orthonormal_basis(order: usize, points: ArrayView2<'_, F>) -> Array2<F> {
        todo!()
    }

    fn grad_orthonormal_basis(order: usize, points: ArrayView2<'_, F>) -> Array2<F> {
        todo!()
    }
}
