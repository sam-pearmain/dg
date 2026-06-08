use std::marker::PhantomData;

use ndarray::{Array1, Array2};

use crate::{float::Float, shapes::{Line, Shape}};

pub trait Rule {}

pub struct GaussLegendreLobatto {}
pub struct GaussLegendre {}

impl Rule for GaussLegendreLobatto {}
impl Rule for GaussLegendre {}

pub trait Quadrature<F: Float, S: Shape<F>> {        
    /// The quadrature points (npoints, ndims)
    fn points(&self, order: usize) -> Array2<F>;
    /// The quadrature weights (npoints)
    fn weights(&self, order: usize) -> Array1<F>;
    /// The quadrature points and weights (npoints, ndims + 1)
    fn points_and_weights(&self, order: usize) -> Array2<F> {
        todo!()
    }
}

pub struct ShapeQuadrature<F: Float, R: Rule, S: Shape<F>> {
    pub order: usize, 
    _marker: PhantomData<(F, R, S)>
}

type GaussLegendreLobattoLine<F: Float> = ShapeQuadrature<F, GaussLegendreLobatto, Line<F>>;

impl<F: Float> Quadrature<F, Line<F>> for GaussLegendreLobattoLine<F> {
    fn points(&self, order: usize) -> Array2<F> {
        todo!()
    }

    fn weights(&self, order: usize) -> Array1<F> {
        todo!()
    }
}