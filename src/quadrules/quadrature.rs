use std::marker::PhantomData;

use ndarray::{Array1, Array2};

use crate::{
    float::Float,
    shapes::{Line, Quad, Shape, Tri},
};

/// The marker trait for quadrature rules
trait QuadratureRule {}

pub struct GaussLegendreLobatto {}
pub struct GaussLegendre {}
pub struct AlphaOpt {}
pub struct WilliamsShunn {}
pub struct ShunnHam {}
pub struct Witherden {}
pub struct WitherdenVincent {}

macro_rules! quadrature_rule {
    ($rule:ty) => {
        impl QuadratureRule for $rule {}
    };
}

quadrature_rule!(GaussLegendreLobatto);
quadrature_rule!(GaussLegendre);
quadrature_rule!(AlphaOpt);
quadrature_rule!(WilliamsShunn);
quadrature_rule!(ShunnHam);
quadrature_rule!(Witherden);
quadrature_rule!(WitherdenVincent);

trait ValidQuadratureRule<F: Float, S: Shape<F>>: QuadratureRule {}

macro_rules! valid_quadrature {
    ($rule:ty, $shape:ty) => {
        impl<F: Float> ValidQuadratureRule<F, $shape> for $rule {}
    };
}

// lines
valid_quadrature!(GaussLegendreLobatto, Line<F>);
valid_quadrature!(GaussLegendre, Line<F>);

// triangles
valid_quadrature!(AlphaOpt, Tri<F>);
valid_quadrature!(WilliamsShunn, Tri<F>);
valid_quadrature!(WitherdenVincent, Tri<F>);

// quadrilaterals
valid_quadrature!(GaussLegendreLobatto, Quad<F>);
valid_quadrature!(GaussLegendre, Quad<F>);
valid_quadrature!(WitherdenVincent, Quad<F>);

/// The quadrature implementation
pub trait Quadrature<F: Float, S: Shape<F>> {
    
    fn rules(&self) -> Vec<QuadratureRule>;
    /// The quadrature points from the number of points
    fn points_from_npoints(&self, npoints: usize) -> Option<Array2<F>>;
    /// The quadrature points required to attain a certain degree of accuracy
    fn points_from_degree(&self, degree: usize) -> Option<Array2<F>>;
    /// The quadrature weights for a given number of quadrature points
    fn weights_from_npoints(&self, npoints: usize) -> Option<Array1<F>>;
    /// The quadrature weights required to attain a certain degree of accuracy
    fn weights_from_degree(&self, degree: usize) -> Option<Array1<F>>;
    /// The points and weights for a given number of quadrature points
    fn points_and_weights_from_npoints(&self, npoints: usize) -> Option<(Array2<F>, Array1<F>)> {
        if let (Some(points), Some(weights)) = (
            self.points_from_npoints(npoints),
            self.weights_from_npoints(npoints),
        ) {
            return Some((points, weights));
        }
        None
    }
    /// The points and weights for a given degree of accuracy
    fn points_and_weights_from_degree(&self, degree: usize) -> Option<(Array2<F>, Array1<F>)> {
        if let (Some(points), Some(weights)) = (
            self.points_from_degree(degree),
            self.weights_from_degree(degree),
        ) {
            return Some((points, weights));
        }
        None
    }
}

pub struct ShapeQuadrature<F: Float, S: Shape<F>, R: ValidQuadratureRule<F, S>> {
    pub order: usize,
    _marker: PhantomData<(F, S, R)>,
}

impl<F: Float, S: Shape<F>, R: ValidQuadratureRule<F, S>> ShapeQuadrature<F, S, R> {
    fn new(order: usize) -> Self {
        Self {
            order,
            _marker: PhantomData,
        }
    }
}
