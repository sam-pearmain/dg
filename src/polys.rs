use std::marker::PhantomData;

use ndarray::{Array2, ArrayView1, ArrayView2, Axis};
use ndarray_linalg::Inverse;

use crate::float::Float;
use crate::shapes::{Line, Shape};

pub trait Basis<F>
where
    F: Float
{
    /// The orthonormal basis at given points
    fn orthonormal_basis_at(&self, points: ArrayView2<'_, F>) -> Array2<F>;
    /// The derivative of the orthonormal basis at given points
    fn grad_orthonormal_basis_at(&self, points: ArrayView2<'_, F>) -> Array2<F>;
    /// The Vandermonde matrix at given points
    fn vandermonde_at(&self, points: ArrayView2<'_, F>) -> Array2<F> {
        self.orthonormal_basis_at(points)
    }
    /// The inverted Vandermonde matrix at given points
    fn inverted_vandermonde_at(&self, points: ArrayView2<'_, F>) -> Array2<F> {
        self.vandermonde_at(points)
            .inv()
            .expect("failed vandermonde inversion, check precision and polynomial order")
    }
    /// The nodal basis at given points
    fn nodal_basis_at(&self, points: ArrayView2<'_, F>) -> Array2<F> {
        self.orthonormal_basis_at(points)
            .dot(&self.inverted_vandermonde_at(points))
    }
    /// The derivative of the nodal basis at given points
    fn grad_nodal_basis_at(&self, points: ArrayView2<'_, F>) -> Array2<F> {
        self.grad_orthonormal_basis_at(points)
            .dot(&self.inverted_vandermonde_at(points))
    }
}

pub struct ReferenceShape<F: Float, S: Shape<F>> {
    order: usize,
    _marker: PhantomData<(F, S)>
}

pub type ReferenceLine<F: Float> = ReferenceShape<F, Line>;

impl<F> Basis<F> for LineBasis
where
    F: Float + Scalar + Lapack,
{
    const SHAPE: Shapes = Shapes::Line;

    fn orthonormal_basis_at(&self, points: ArrayView2<'_, F>) -> Array2<F> {
        let half = F::from(0.5).unwrap();
        let mut basis = legendre(self.order, points.column(0));

        for (i, mut col) in basis.axis_iter_mut(Axis(1)).enumerate() {
            let i_as_float = F::from(i).unwrap();
            col.mapv_inplace(|val| val * Float::sqrt(i_as_float + half));
        }

        basis
    }

    fn grad_orthonormal_basis_at(&self, points: ArrayView2<'_, F>) -> Array2<F> {
        let half = F::from(0.5).unwrap();
        let mut dbasis = legendre_derivative(self.order, points.column(0));

        for (i, mut col) in dbasis.axis_iter_mut(Axis(1)).enumerate() {
            let i_as_float = F::from(i).unwrap();
            col.mapv_inplace(|val| val * Float::sqrt(i_as_float + half));
        }

        dbasis
    }
}

pub struct QuadBasis {
    order: usize,
}

impl<F> Basis<F> for QuadBasis
where
    F: Float + Scalar + Lapack,
{
    const SHAPE: Shapes = Shapes::Quadrilateral;

    fn orthonormal_basis_at(&self, points: ArrayView2<'_, F>) -> Array2<F> {
        unimplemented!()
    }

    fn grad_orthonormal_basis_at(&self, points: ArrayView2<'_, F>) -> Array2<F> {
        unimplemented!()
    }
}

#[rustfmt::skip]
/// Computes the value of the Jacobi polynomials at specified evaluation points to a given degree
pub fn jacobi<F: Float>(
    degree: usize, 
    alpha: F, 
    beta: F, 
    points: ArrayView1<'_, F>
) -> Array2<F> {
    let one = F::one();
    let two = F::from(2).unwrap();

    let mut j = Array2::<F>::ones((points.len(), degree + 1));

    if degree >= 1 {
        for i in 0..j.nrows() {
            j[[i, 1]] = ((alpha + beta + two) * points[i] + (alpha - beta)) / two;
        } 
    }

    if degree >= 2 {
        for deg in 2..=degree {
            let deg_as_float = F::from(deg).unwrap();

            let aq = (alpha + beta + two * deg_as_float)
                * (alpha + beta + two * deg_as_float - one)
                / (two * deg_as_float * (alpha + beta + deg_as_float));

            let bq = (alpha + beta + two * deg_as_float - one) * (beta * beta - alpha * alpha)
                / (two
                    * deg_as_float
                    * (alpha + beta + deg_as_float)
                    * (alpha + beta + two * deg_as_float - two));

            let cq = (alpha + beta + two * deg_as_float)
                * (alpha + deg_as_float - one)
                * (beta + deg_as_float - one)
                / (deg_as_float
                    * (alpha + beta + deg_as_float)
                    * (alpha + beta + two * deg_as_float - two));

            for i in 0..j.nrows() {
                j[[i, deg]] = (aq * points[i] - bq) * j[[i, deg - 1]] - cq * j[[i, deg - 2]];
            }
        }
    }

    j
}

/// Computes the derivative of the Jacobi polynomials at specified evaluation points to a given degree
pub fn jacobi_derivative<F: Float>(
    degree: usize,
    alpha: F,
    beta: F,
    points: ArrayView1<'_, F>,
) -> Array2<F> {
    let one = F::one();
    let two = F::from(2).unwrap();

    let mut dj = Array2::zeros((points.len(), degree + 1));

    if degree >= 1 {
        let j_previous = jacobi(degree - 1, alpha + one, beta + one, points);

        for deg in 1..degree {
            let deg_as_float = F::from(deg).unwrap();

            for i in 0..points.len() {
                dj[[i, deg]] = j_previous[[i, deg - 1]] * (deg_as_float + alpha + beta + one) / two;
            }
        }
    }

    dj
}

/// Computes the Legendre polynomials at specified points up to a given order
pub fn legendre<F: Float>(degree: usize, points: ArrayView1<'_, F>) -> Array2<F> {
    jacobi(degree, F::zero(), F::zero(), points)
}

/// Computes the derivative of the Legendre polynomials at specified points up to a given order
pub fn legendre_derivative<F: Float>(degree: usize, points: ArrayView1<'_, F>) -> Array2<F> {
    jacobi_derivative(degree, F::zero(), F::zero(), points)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    macro_rules! assert_array_approx_eq {
        ($alpha:expr, $beta:expr, $tol:expr) => {
            assert_eq!($alpha.len(), $beta.len(), "arrays have different length");
            for (v1, v2) in $alpha.iter().zip($beta.iter()) {
                assert!(
                    (*v1 - *v2).abs() < $tol,
                    "values differ by more than tolerance: {} != {}",
                    v1,
                    v2
                );
            }
        };
    }

    #[test]
    fn test_legendre_polynomials() {
        let points = array![-1.0_f64, 0.0_f64, 1.0_f64];
        let degree = 2;
        let j = jacobi(degree, 0.0, 0.0, points.view());

        assert_array_approx_eq!(j.row(0), array![1.0, 1.0, 1.0], 1e-14);
        assert_array_approx_eq!(j.row(1), array![-1.0, 0.0, 1.0], 1e-14);
        assert_array_approx_eq!(j.row(2), array![1.0, -0.5, 1.0], 1e-14);
    }

    #[test]
    fn test_legendre_derivatives() {
        let points = array![-1.0_f64, 0.0_f64, 1.0_f64];
        let degree = 2;
        let dj = jacobi_derivative(degree, 0.0, 0.0, points.view());

        assert_array_approx_eq!(dj.row(0), array![0.0, 0.0, 0.0], 1e-14);
        assert_array_approx_eq!(dj.row(1), array![1.0, 1.0, 1.0], 1e-14);
        assert_array_approx_eq!(dj.row(2), array![-3.0, 0.0, 3.0], 1e-14);
    }

    #[test]
    fn test_shifted_jacobi_weights() {
        let points = array![-0.5_f64, 0.5_f64];
        let degree = 1;
        let j = jacobi(degree, 1.0, 2.0, points.view());

        assert_array_approx_eq!(j.row(0), array![1.0, 1.0], 1e-14);
        assert_array_approx_eq!(j.row(1), array![-1.75, 0.75], 1e-14);
    }

    #[test]
    fn test_degree_zero_base_case() {
        let points = array![0.33_f64, 0.77_f64];
        let j = jacobi(0, 0.0, 0.0, points.view());
        let dj = jacobi_derivative(0, 0.0, 0.0, points.view());

        assert_eq!(j.shape(), &[1, 2]);
        assert_array_approx_eq!(j.row(0), array![1.0, 1.0], 1e-14);
        assert_eq!(dj.shape(), &[1, 2]);
        assert_array_approx_eq!(dj.row(0), array![0.0, 0.0], 1e-14);
    }

    #[test]
    fn test_vandermonde_inversion() {
        use ndarray_linalg::Inverse;

        let points = array![-1.0_f32, 0.0_f32, 1.0_f32];
        let degree = 2;

        let j = jacobi(degree, 0.0, 0.0, points.view());
        let v = j.t().to_owned();
        let v_inv = v.inv().expect("vandermonde matrix inversion failed");
        let identity = v.dot(&v_inv);

        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (identity[[i, j]] - expected).abs() < 1e-13,
                    "identity matrix mismatch at [{}, {}]: got {}, expected {}",
                    i,
                    j,
                    identity[[i, j]],
                    expected
                );
            }
        }
    }
}
