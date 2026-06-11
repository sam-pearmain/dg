pub mod line;
pub mod quadrature;

#[macro_export]
macro_rules! quadrule_impl {
    (rule: $rule:ident, shape: $shape:ident, degree: $deg:expr, points: $pts:tt, weights: $wts:tt) => {
        ::paste::paste! {
            pub struct [<$rule $shape D $deg>]<F: $crate::float::Float> {
                points: ::std::sync::OnceLock<::ndarray::Array2<F>>,
                weights: ::std::sync::OnceLock<::ndarray::Array1<F>>,
            }

            impl<F: $crate::float::Float> [<$rule $shape D $deg>]<F> {
                pub fn new() -> Self {
                    Self {
                        points: ::std::sync::OnceLock::new(),
                        weights: ::std::sync::OnceLock::new(),
                    }
                }
            }

            impl<F: $crate::float::Float> $crate::quadrules::quadrature::QuadratureRule<F>
                for [<$rule $shape D $deg>]<F>
            {
                fn shape(&self) -> $crate::shapes::ShapeFamily {
                    $crate::shapes::ShapeFamily::$shape
                }

                fn family(&self) -> $crate::quadrules::quadrature::QuadratureFamily {
                    $crate::quadrules::quadrature::QuadratureFamily::$rule
                }

                fn points(&self) -> ::ndarray::ArrayView2<'_, F> {
                    self.points.get_or_init(|| $crate::arrayf!(F, $pts)).view()
                }

                fn weights(&self) -> ::ndarray::ArrayView1<'_, F> {
                    self.weights.get_or_init(|| $crate::arrayf!(F, $wts)).view()
                }

                fn degree(&self) -> usize {
                    $deg
                }
            }
        }
    };
}