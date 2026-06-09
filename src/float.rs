use cudarc::driver::DeviceRepr;
use ndarray_linalg::{Lapack, Scalar};
use num_traits::Float as NumTraitsFloat;

macro_rules! f {
    ($F:ty, $number:ident) => {
        <$F>::from($number).unwrap()
    };
}

pub trait Float: NumTraitsFloat + Lapack + Scalar + DeviceRepr {
    const C_TYPE_STRING: &'static str;
}

impl Float for f32 {
    const C_TYPE_STRING: &'static str = "float";
}

impl Float for f64 {
    const C_TYPE_STRING: &'static str = "double";
}
