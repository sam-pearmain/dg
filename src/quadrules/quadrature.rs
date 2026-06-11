use std::marker::PhantomData;

use anyhow::Result;
use ndarray::{ArrayView1, ArrayView2};

use crate::{
    float::Float,
    shapes::{Hex, Line, Pri, Pyr, Quad, Shape, ShapeFamily, Tet, Tri},
};

/// A quadrature rule
pub trait QuadratureRule<F: Float> {
    /// The shape on which the quadrature is defined
    fn shape(&self) -> ShapeFamily;
    /// The quadrature family
    fn family(&self) -> QuadratureType;
    /// The quadrature points
    fn points(&self) -> ArrayView2<'_, F>;
    /// The quadrature weights
    fn weights(&self) -> ArrayView1<'_, F>;
    /// The degree of accuracy
    fn degree(&self) -> usize;
    /// The number of quadrature points
    fn num_quadrature_points(&self) -> usize {
        self.points().len()
    }
}

/// A factory trait for quadrature rules for of a given shape
pub trait QuadratureFactory<F: Float, S: Shape<F>> {
    /// The quadrature rule constructor
    fn from_rule_and_degree(
        rule: QuadratureType,
        degree: usize,
    ) -> Result<Box<dyn QuadratureRule<F>>>;
}

#[derive(Debug, Clone)]
pub enum QuadratureType {
    GaussLegendreLobatto,
    GaussLegendre,
    WilliamsShunn,
    ShunnHam,
    Witherden,
    WitherdenVincent,
    WilliamsShunnGaussLegendreLobatto,
}

/// The quadrature factory
pub struct Quadrature<F: Float, S: Shape<F>> {
    _marker: PhantomData<(F, S)>,
}

pub type LineQuadrature<F> = Quadrature<F, Line<F>>;
pub type TriQuadrature<F> = Quadrature<F, Tri<F>>;
pub type QuadQuadrature<F> = Quadrature<F, Quad<F>>;
pub type TetQuadrature<F> = Quadrature<F, Tet<F>>;
pub type HexQuadrature<F> = Quadrature<F, Hex<F>>;
pub type PriQuadrature<F> = Quadrature<F, Pri<F>>;
pub type PyrQuadrature<F> = Quadrature<F, Pyr<F>>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test() {
        let quadrule =
            LineQuadrature::<f64>::from_rule_and_degree(QuadratureType::GaussLegendreLobatto, 38)
                .unwrap();
        println!("{}", quadrule.points())
    }
}
