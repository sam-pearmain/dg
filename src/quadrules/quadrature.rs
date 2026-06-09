use std::marker::PhantomData;

use anyhow::{Result, anyhow};
use ndarray::{Array1, Array2};

use crate::{
    float::Float,
    shapes::{Hex, Line, Quad, Shape, Tet, Tri},
};

pub trait QuadratureDegree {
    /// The degree of accuracy for the specific quadrature rule
    fn degree(&self) -> usize;
}

/// The top-level quadrature rule trait
pub trait QuadratureRule<F: Float>: QuadratureDegree {
    /// The quadrature points
    fn points(&self) -> Array2<F>;
    /// The quadrature weights
    fn weights(&self) -> Option<Array1<F>>;
}

/// A marker trait for quadrature family
pub trait QuadratureFamily {}

macro_rules! quadrature_families {
    ( $( $family:ident ),* ) => {
        $(
            #[derive(Debug, Clone)]
            pub struct $family {}

            impl QuadratureFamily for $family {}
        )*
    };
}

quadrature_families!(
    GaussLegendreLobatto,
    GaussLegendre,
    AlphaOpt,
    WilliamsShunn,
    ShunnHam,
    Witherden,
    WitherdenVincent
);

#[derive(Debug, Clone)]
pub enum QuadratureKind {
    GaussLegendreLobatto(GaussLegendreLobatto),
    GaussLegendre(GaussLegendre),
    AlphaOpt(AlphaOpt),
    WilliamsShunn(WilliamsShunn),
    ShunnHam(ShunnHam),
    Witherden(Witherden),
    WitherdenVincent(WitherdenVincent),
}

/// A marker trait for valid combinations of shapes with quadrature rules
pub trait ValidQuadratureRule<F: Float, S: Shape<F>>: QuadratureFamily {}

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

// tetrahedrons
valid_quadrature!(AlphaOpt, Tet<F>);
valid_quadrature!(ShunnHam, Tet<F>);
valid_quadrature!(Witherden, Tet<F>);
valid_quadrature!(WitherdenVincent, Tet<F>);

// hexahedrons
valid_quadrature!(GaussLegendreLobatto, Hex<F>);
valid_quadrature!(GaussLegendre, Hex<F>);
valid_quadrature!(Witherden, Hex<F>);
valid_quadrature!(WitherdenVincent, Hex<F>);

#[macro_export]
macro_rules! quadrule_impl {
    ($alias:ident, points: $pts:tt, weights: $wts:tt) => {
        impl<F: $crate::float::Float> $crate::quadrules::quadrature::QuadratureRule<F> for $alias<F> {
            fn points(&self) -> ::ndarray::Array2<F> {
                $crate::arrayf!(F, $pts)
            }

            fn weights(&self) -> Option<::ndarray::Array1<F>> {
                Some($crate::arrayf!(F, $wts))
            }
        }
    };
    ($alias:ident, points: $pts:tt, weights: None) => {
        impl<F: $crate::float::Float> $crate::quadrules::quadrature::QuadratureRule<F> for $alias<F> {
            fn points(&self) -> ::ndarray::Array2<F> {
                $crate::arrayf!(F, $pts)
            }

            fn weights(&self) -> Option<::ndarray::Array1<F>> {
                None
            }
        }
    };
}

pub struct ShapeQuadrature<
    F: Float,
    S: Shape<F>,
    R: ValidQuadratureRule<F, S>,
    const D: usize,
    const N: usize,
> {
    _marker: PhantomData<(F, S, R)>,
}

impl<F, S, R, const D: usize, const N: usize> ShapeQuadrature<F, S, R, D, N>
where
    F: Float,
    S: Shape<F>,
    R: ValidQuadratureRule<F, S>,
{
    pub const fn new() -> Self {
        Self {
            _marker: PhantomData,
        }
    }
}

impl<F, S, R, const D: usize, const N: usize> QuadratureDegree for ShapeQuadrature<F, S, R, D, N>
where
    F: Float,
    S: Shape<F>,
    R: ValidQuadratureRule<F, S>,
{
    fn degree(&self) -> usize {
        D
    }
}

impl<F: Float + 'static> dyn QuadratureRule<F> {
    pub fn line(rule: QuadratureKind, degree: usize) -> Result<Box<Self>> {
        match rule {
            QuadratureKind::GaussLegendreLobatto(_) => {
                todo!()
            }
            QuadratureKind::GaussLegendre(_) => {
                todo!()
            }
            _ => Err(anyhow!(format!("{rule:?} unsupported over line elements"))),
        }
    }
}
