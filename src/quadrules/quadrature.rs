use std::marker::PhantomData;

use anyhow::{Ok, Result, anyhow};
use ndarray::{ArrayView1, ArrayView2};

use crate::{
    float::Float,
    quadrules::line::{GaussLegendreLobattoLineD1, GaussLegendreLobattoLineD3, GaussLegendreLobattoLineD5, GaussLegendreLobattoLineD7, GaussLegendreLobattoLineD9, GaussLegendreLobattoLineD11, GaussLegendreLobattoLineD13, GaussLegendreLobattoLineD15, GaussLegendreLobattoLineD17, GaussLegendreLobattoLineD19, GaussLegendreLobattoLineD21, GaussLegendreLobattoLineD23, GaussLegendreLobattoLineD25, GaussLegendreLobattoLineD27, GaussLegendreLobattoLineD29, GaussLegendreLobattoLineD31, GaussLegendreLobattoLineD33, GaussLegendreLobattoLineD35, GaussLegendreLobattoLineD37},
    shapes::ShapeFamily,
};

/// A quadrature rule
pub trait QuadratureRule<F: Float> {
    /// The shape on which the quadrature is defined
    fn shape(&self) -> ShapeFamily;
    /// The quadrature family
    fn family(&self) -> QuadratureFamily;
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

#[derive(Debug, Clone)]
pub enum QuadratureFamily {
    GaussLegendreLobatto,
    GaussLegendre,
    WilliamsShunn,
    ShunnHam,
    Witherden,
    WitherdenVincent,
    WilliamsShunnGaussLegendreLobatto,
}

/// The quadrature factory
pub struct Quadrature<F: Float> {
    _marker: PhantomData<F>,
}

impl<F: Float> Quadrature<F> {
    pub fn from_rule_shape_and_degree(rule: QuadratureFamily, shape: ShapeFamily, degree: usize) -> Result<Box<dyn QuadratureRule<F>>> {
        let incompatible = anyhow!("incompatible quadrature rule and shape combination:\nquadrule: {rule:?}\nshape: {shape:?}");
        let unsuitable = anyhow!("no suitable quadrature rule found to satisfy degree of accuracy: {degree:?}\nquadrule: {rule:?}, shape: {shape:?}");

        match shape {
            ShapeFamily::Line => match rule {
                QuadratureFamily::GaussLegendreLobatto => match degree {
                    0..=1 => Ok(Box::new(GaussLegendreLobattoLineD1::new())), 
                    2..=3 => Ok(Box::new(GaussLegendreLobattoLineD3::new())), 
                    4..=5 => Ok(Box::new(GaussLegendreLobattoLineD5::new())),
                    6..=7 => Ok(Box::new(GaussLegendreLobattoLineD7::new())),
                    8..=9 => Ok(Box::new(GaussLegendreLobattoLineD9::new())),
                    10..=11 => Ok(Box::new(GaussLegendreLobattoLineD11::new())), 
                    12..=13 => Ok(Box::new(GaussLegendreLobattoLineD13::new())), 
                    14..=15 => Ok(Box::new(GaussLegendreLobattoLineD15::new())), 
                    16..=17 => Ok(Box::new(GaussLegendreLobattoLineD17::new())), 
                    18..=19 => Ok(Box::new(GaussLegendreLobattoLineD19::new())), 
                    20..=21 => Ok(Box::new(GaussLegendreLobattoLineD21::new())),
                    22..=23 => Ok(Box::new(GaussLegendreLobattoLineD23::new())),
                    24..=25 => Ok(Box::new(GaussLegendreLobattoLineD25::new())),
                    26..=27 => Ok(Box::new(GaussLegendreLobattoLineD27::new())), 
                    28..=29 => Ok(Box::new(GaussLegendreLobattoLineD29::new())), 
                    30..=31 => Ok(Box::new(GaussLegendreLobattoLineD31::new())), 
                    32..=33 => Ok(Box::new(GaussLegendreLobattoLineD33::new())), 
                    34..=35 => Ok(Box::new(GaussLegendreLobattoLineD35::new())), 
                    36..=37 => Ok(Box::new(GaussLegendreLobattoLineD37::new())),
                    _ => Err(unsuitable)
                }, 
                QuadratureFamily::GaussLegendre => match degree {
                    0..=1 => Ok(Box::new(GaussLegendreLineD1::new())), 
                    2..=3 => Ok(Box::new(GaussLegendreLineD3::new())), 
                    4..=5 => Ok(Box::new(GaussLegendreLineD5::new())),
                    6..=7 => Ok(Box::new(GaussLegendreLineD7::new())),
                    8..=9 => Ok(Box::new(GaussLegendreLineD9::new())),
                    10..=11 => Ok(Box::new(GaussLegendreLineD11::new())), 
                    12..=13 => Ok(Box::new(GaussLegendreLineD13::new())), 
                    14..=15 => Ok(Box::new(GaussLegendreLineD15::new())), 
                    16..=17 => Ok(Box::new(GaussLegendreLineD17::new())), 
                    18..=19 => Ok(Box::new(GaussLegendreLineD19::new())), 
                    20..=21 => Ok(Box::new(GaussLegendreLineD21::new())),
                    22..=23 => Ok(Box::new(GaussLegendreLineD23::new())),
                    24..=25 => Ok(Box::new(GaussLegendreLineD25::new())),
                    26..=27 => Ok(Box::new(GaussLegendreLineD27::new())), 
                    28..=29 => Ok(Box::new(GaussLegendreLineD29::new())), 
                    30..=31 => Ok(Box::new(GaussLegendreLineD31::new())), 
                    32..=33 => Ok(Box::new(GaussLegendreLineD33::new())), 
                    34..=35 => Ok(Box::new(GaussLegendreLineD35::new())), 
                    36..=37 => Ok(Box::new(GaussLegendreLineD37::new())),
                }
                _ => Err(incompatible)
            }, 
            _ => todo!()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test() {
        let quadrule = Quadrature::<f64>::from_rule_shape_and_degree(
            QuadratureFamily::GaussLegendreLobatto, 
            ShapeFamily::Line, 
            2
        ).unwrap();
        println!("{}", quadrule.points())
    }
}
