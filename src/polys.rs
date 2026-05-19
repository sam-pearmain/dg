use ndarray::{Array2, ArrayView1};
use num_traits::Float;

pub trait Basis<F>
where
    F: Float,
{
    fn orthonormal_basis_at(points: ArrayView1<'_, F>) -> Array2<F>;
}

/// Computes the value of the jacobi polynomials at specified evaluation points to a given degree
pub fn jacobi<F: Float>(degree: usize, alpha: F, beta: F, points: ArrayView1<'_, F>) -> Array2<F> {
    let one = F::one();
    let two = F::from(2).unwrap();

    let mut j = Array2::<F>::ones((degree + 1, points.len()));

    if degree >= 1 {
        let j1 = points.mapv(|zi| ((alpha + beta + two) * zi + (alpha - beta)) / two);
        j.row_mut(1).assign(&j1);
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

            for i in 0..points.len() {
                let val = (aq * points[i] - bq) * j[[deg - 1, i]] - cq * j[[deg - 2, i]];
                j[[deg, i]] = val;
            }
        }
    }

    j
}

/// Computes the derivative of the jacobi polynomials at specified evaluation points to a given degree
pub fn jacobi_derivative<F: Float>(
    degree: usize,
    alpha: F,
    beta: F,
    points: ArrayView1<'_, F>,
) -> Array2<F> {
    let one = F::one();
    let two = F::from(2).unwrap();

    let mut dj = Array2::zeros((degree + 1, points.len()));

    if degree >= 1 {
        let j_previous = jacobi(degree - 1, alpha + one, beta + one, points);

        for deg in 0..degree {
            let deg_as_float = F::from(deg).unwrap();

            for i in 0..points.len() {
                dj[[deg + 1, i]] = j_previous[[deg, i]] * (deg_as_float + alpha + beta + two) / two;
            }
        }
    }

    dj
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

        let points = array![-1.0_f64, 0.0_f64, 1.0_f64];
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
