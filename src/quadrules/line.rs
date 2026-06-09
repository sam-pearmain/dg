use ndarray::{Array1, Array2};

use crate::{
    float::Float,
    quadrules::quadrature::{GaussLegendre, GaussLegendreLobatto, Quadrature, ShapeQuadrature},
    shapes::Line,
};

struct GaussLegendreLobattoLine<const DEGREE: usize, const NPOINTS: usize>;

#[allow(type_alias_bounds)]
type GaussLegendreLobattoLine<F: Float> = ShapeQuadrature<F, Line<F>, GaussLegendreLobatto>;

impl<F: Float> Quadrature<F, Line<F>> for GaussLegendreLobattoLine<F> {
    fn points(&self, order: usize) -> Array2<F> {
        todo!()
    }

    fn weights(&self, order: usize) -> Option<Array1<F>> {
        todo!()
    }
}

#[allow(type_alias_bounds)]
type GaussLegendreLine<F: Float> = ShapeQuadrature<F, Line<F>, GaussLegendre>;
