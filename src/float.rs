use cudarc::driver::DeviceRepr;
use ndarray_linalg::{Lapack, Scalar};
use num_traits::Float as NumTraitsFloat;

#[macro_export]
macro_rules! f {
    ($float:literal) => {
        $crate::float::Float::from_f64($float)
    };
}

#[macro_export]
macro_rules! arrayf {
    ($F:ty, [ $( [ $($x:expr),* $(,)? ] ),* $(,)? ]) => {
        ::ndarray::array![
            $(
                [ $( $crate::f!($x) ),* ]
            ),*
        ]
    };
    ($F:ty, [ $($x:expr),* $(,)? ]) => {
        ::ndarray::array![
            $( $crate::f!($x) ),*
        ]
    };
}

pub trait Float: NumTraitsFloat + Lapack + Scalar + DeviceRepr {
    const C_TYPE_STRING: &'static str;

    fn from_f64(val: f64) -> Self;
}

impl Float for f32 {
    const C_TYPE_STRING: &'static str = "float";
    
    #[inline(always)]
    fn from_f64(val: f64) -> Self {
        val as f32
    }
}

impl Float for f64 {
    const C_TYPE_STRING: &'static str = "double";
    
    #[inline(always)]
    fn from_f64(val: f64) -> Self {
        val
    }
}
