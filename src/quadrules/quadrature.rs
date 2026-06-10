use std::marker::PhantomData;

use anyhow::{Result, anyhow};
use ndarray::{Array1, Array2};

use crate::{
    float::Float,
    shapes::{Hex, Line, Pri, Pyr, Quad, Shape, Tet, Tri},
};

/// The quadrature metadata
pub trait Quadrature {
    /// The degree of accuracy
    fn degree(&self) -> usize;
    /// The number of quadrature points
    fn num_quadrature_points(&self) -> usize;
}

/// The top-level quadrature trait
pub trait QuadratureRule<F: Float>: Quadrature {
    /// The quadrature points
    fn points(&self) -> Array2<F>;
    /// The quadrature weights
    fn weights(&self) -> Option<Array1<F>>;
}

/// A marker for tensor-product quadrature rules
pub trait TensorProductQuadratureRule {}

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
    WitherdenVincent,
    AlphaOptGaussLegendreLobatto,
    WilliamsShunnGaussLegendreLobatto
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
    AlphaOptGaussLegendreLobatto(AlphaOptGaussLegendreLobatto),
    WilliamsShunnGaussLegendreLobatto(WilliamsShunnGaussLegendreLobatto),
}

/// A marker trait for valid combinations of shapes with quadrature rules
pub trait ValidQuadratureRule<F: Float, S: Shape<F>>: QuadratureFamily {}

macro_rules! valid_quadrature {
    ($rule:ident, $shape:ident) => {
        impl<F: Float> ValidQuadratureRule<F, $shape<F>> for $rule {}
    };
}

// lines
valid_quadrature!(GaussLegendreLobatto, Line);
valid_quadrature!(GaussLegendre, Line);

// triangles
valid_quadrature!(AlphaOpt, Tri);
valid_quadrature!(WilliamsShunn, Tri);
valid_quadrature!(WitherdenVincent, Tri);

// quadrilaterals
valid_quadrature!(GaussLegendreLobatto, Quad);
valid_quadrature!(GaussLegendre, Quad);
valid_quadrature!(WitherdenVincent, Quad);

// tetrahedrons
valid_quadrature!(AlphaOpt, Tet);
valid_quadrature!(ShunnHam, Tet);
valid_quadrature!(Witherden, Tet);
valid_quadrature!(WitherdenVincent, Tet);

// hexahedrons
valid_quadrature!(GaussLegendreLobatto, Hex);
valid_quadrature!(GaussLegendre, Hex);
valid_quadrature!(Witherden, Hex);
valid_quadrature!(WitherdenVincent, Hex);

// prisms
valid_quadrature!(AlphaOptGaussLegendreLobatto, Pri);
valid_quadrature!(WilliamsShunnGaussLegendreLobatto, Pri);
valid_quadrature!(Witherden, Pri);
valid_quadrature!(WitherdenVincent, Pri);

// pyramids
valid_quadrature!(GaussLegendreLobatto, Pyr);
valid_quadrature!(GaussLegendre, Pyr);
valid_quadrature!(Witherden, Pyr);
valid_quadrature!(WitherdenVincent, Pyr);

#[macro_export]
macro_rules! quadrule_impl {
    ($alias:ident, points: $pts:tt, weights: $wts:tt) => {        
        impl<F: $crate::float::Float> $crate::quadrules::quadrature::QuadratureRule<F>
            for $alias<F>
        {
            fn points(&self) -> ::ndarray::Array2<F> {
                $crate::arrayf!(F, $pts)
            }

            fn weights(&self) -> Option<::ndarray::Array1<F>> {
                Some($crate::arrayf!(F, $wts))
            }
        }
    };
    ($alias:ident, points: $pts:tt, weights: None) => {
        impl<F: $crate::float::Float> $crate::quadrules::quadrature::QuadratureRule<F>
            for $alias<F>
        {
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

impl<F, S, R, const D: usize, const N: usize> Quadrature for ShapeQuadrature<F, S, R, D, N>
where
    F: Float,
    S: Shape<F>,
    R: ValidQuadratureRule<F, S>,
{
    fn degree(&self) -> usize {
        D
    }
    
    fn num_quadrature_points(&self) -> usize {
        N
    }
}

impl<F: Float + 'static> dyn QuadratureRule<F> {
    pub fn line(rule: QuadratureKind, degree: usize) -> Result<Box<Self>> {
        match rule {
            QuadratureKind::GaussLegendreLobatto(_) => match degree {
                _ => todo!(),
            },
            QuadratureKind::GaussLegendre(_) => match degree {
                _ => todo!(),
            },
            _ => Err(anyhow!(format!("{rule:?} unsupported over line elements"))),
        }
    }
}
