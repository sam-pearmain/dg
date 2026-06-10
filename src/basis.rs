use std::marker::PhantomData;

use ndarray::{Array2, Array3, ArrayView2, s};
use ndarray_linalg::Inverse;

use crate::float::Float;
use crate::polys::{legendre, legendre_derivative};
use crate::shapes::{Dimensioned, Hex, Line, Pri, Pyr, Quad, Shape, Tet, Tri};

pub trait Basis<F>
where
    F: Float,
{
    type Shape: Shape<F>;
    /// The orthonormal basis set at given points (npoints, nmodes)
    fn orthonormal_basis(order: usize, points: ArrayView2<'_, F>) -> Array2<F>;
    /// The derivatives of the orthonormal basis at given points (npoints, nmodes, ndims)
    fn grad_orthonormal_basis(order: usize, points: ArrayView2<'_, F>) -> Array3<F>;
    /// The Vandermonde matrix at given points (npoints, nmodes)
    fn vandermonde(order: usize, nodals: ArrayView2<'_, F>) -> Array2<F> {
        Self::orthonormal_basis(order, nodals)
    }
    /// The inverted Vandermonde matrix at given points (npoints, nmodes)
    fn inverted_vandermonde(order: usize, nodals: ArrayView2<'_, F>) -> Array2<F> {
        Self::vandermonde(order, nodals)
            .inv()
            .expect("failed vandermonde inversion, check precision and polynomial order")
    }
    /// The nodal basis at given points (npoints, nmodes)
    fn nodal_basis(
        order: usize,
        points: ArrayView2<'_, F>,
        nodals: ArrayView2<'_, F>,
    ) -> Array2<F> {
        Self::orthonormal_basis(order, points).dot(&Self::inverted_vandermonde(order, nodals))
    }
    /// The derivatives of the nodal basis at given points (npoints, nmodes, ndims)
    fn grad_nodal_basis(
        order: usize,
        points: ArrayView2<'_, F>,
        nodals: ArrayView2<'_, F>,
    ) -> Array3<F> {
        let ortho_dbasis = Self::grad_orthonormal_basis(order, points);
        let inverted_vandermonde = Self::inverted_vandermonde(order, nodals);
        let mut nodal_dbasis = Array3::zeros((
            ortho_dbasis.shape()[0],
            inverted_vandermonde.shape()[1],
            ortho_dbasis.shape()[2],
        ));

        for dim in 0..ortho_dbasis.shape()[2] {
            nodal_dbasis.slice_mut(s![.., .., dim]).assign(
                &ortho_dbasis
                    .slice(s![.., .., dim])
                    .dot(&inverted_vandermonde),
            );
        }

        nodal_dbasis
    }
}

/// The set of basis functions defined over a specific shape
pub struct ShapeBasis<F: Float, S: Shape<F>> {
    _marker: PhantomData<(F, S)>,
}

/// todo
pub struct BasisCache<F: Float> {
    _marker: PhantomData<F>,
}

/// todo
pub struct CachedShapeBasis<F: Float, S: Shape<F>> {
    basis: ShapeBasis<F, S>,
    cache: BasisCache<F>,
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
    type Shape = Line<F>;

    fn orthonormal_basis(order: usize, points: ArrayView2<'_, F>) -> Array2<F> {
        let half = F::from(0.5).unwrap();
        let mut basis = Array2::zeros((points.nrows(), order + 1));

        for n in 0..=order {
            let n_as_float = F::from(n).unwrap();
            let scale_factor = num_traits::Float::sqrt(n_as_float + half);
            let mut col = basis.column_mut(n);
            col.assign(&legendre(n, points.column(0)));
            col.mapv_inplace(|val| val * scale_factor);
        }

        basis
    }

    fn grad_orthonormal_basis(order: usize, points: ArrayView2<'_, F>) -> Array3<F> {
        let half = F::from(0.5).unwrap();
        let mut dbasis = Array3::zeros((points.nrows(), order + 1, Self::Shape::ndims()));

        for n in 0..=order {
            let n_as_float = F::from(n).unwrap();
            let scale_factor = num_traits::Float::sqrt(n_as_float + half);
            let mut col = dbasis.slice_mut(s![.., n, 0]);
            col.assign(&legendre_derivative(n, points.column(0)));
            col.mapv_inplace(|val| val * scale_factor);
        }

        dbasis
    }
}
