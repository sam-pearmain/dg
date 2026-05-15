use ndarray::{Array2, ArrayView1};
use num_traits::Float;

pub fn jacobi<F: Float>(n: usize, a: F, b: F, z: ArrayView1<'_, F>) -> Array2<F> {
    let one = F::one();
    let two = F::from(2).unwrap();
    
    let mut j = Array2::<F>::ones((n + 1, z.len()));

    if n >= 1 {
        let j1 = z.mapv(|zi| ((a + b + two) * zi + (a - b)) / two);
        j.row_mut(1).assign(&j1);
    }

    if n >= 2 {
        for q in 2..=n {
            let q_float = F::from(q).unwrap();

            let aq = (a + b + two * q_float) * (a + b + two * q_float - one) 
                / (two * q_float * (a + b + q_float));

            let bq = (a + b + two * q_float - one) * (b * b - a * a)
                / (two * q_float * (a + b + q_float) * (a + b + two * q_float - two));

            let cq = (a + b + two * q_float) * (a + q_float - one) * (b + q_float - one)
                / (q_float * (a + b + q_float) * (a + b + two * q_float - two));

            for i in 0..z.len() {
                let val = (aq * z[i] - bq) * j[[q - 1, i]] - cq * j[[q - 2, i]];
                j[[q, i]] = val;
            }
        }
    }

    j
}

pub fn jacobi_derivative<F: Float>(n: usize, a: F, b: F, z: ArrayView1<'_, F>) -> Array2<F> {
    let one = F::one();
    let two = F::from(2).unwrap();

    let mut dj = Array2::zeros((n + 1, z.len()));

    if n >= 1 {
        let j_previous = jacobi(n - 1, a + one, b + one, z);
        
        for i in 0..n {
            let i_float = F::from(i).unwrap();
            
            for k in 0..z.len() {
                dj[[i + 1, k]] = j_previous[[i, k]] * (i_float + a + b + two) / two;
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
        ($a:expr, $b:expr, $tol:expr) => {
            assert_eq!($a.len(), $b.len(), "arrays have different length");
            for (v1, v2) in $a.iter().zip($b.iter()) {
                assert!(
                    (*v1 - *v2).abs() < $tol,
                    "values differ by more than tolerance: {} != {}",
                    v1, v2
                );
            }
        };
    }

    #[test]
    fn test_legendre_polynomials() {
        let z = array![-1.0_f64, 0.0_f64, 1.0_f64];
        let n = 2;
        let j = jacobi(n, 0.0, 0.0, z.view());

        assert_array_approx_eq!(j.row(0), array![1.0, 1.0, 1.0], 1e-14);
        assert_array_approx_eq!(j.row(1), array![-1.0, 0.0, 1.0], 1e-14);
        assert_array_approx_eq!(j.row(2), array![1.0, -0.5, 1.0], 1e-14);
    }

    #[test]
    fn test_legendre_derivatives() {
        let z = array![-1.0_f64, 0.0_f64, 1.0_f64];
        let n = 2;
        let dj = jacobi_derivative(n, 0.0, 0.0, z.view());

        assert_array_approx_eq!(dj.row(0), array![0.0, 0.0, 0.0], 1e-14);
        assert_array_approx_eq!(dj.row(1), array![1.0, 1.0, 1.0], 1e-14);
        assert_array_approx_eq!(dj.row(2), array![-3.0, 0.0, 3.0], 1e-14);
    }

    #[test]
    fn test_shifted_jacobi_weights() {
        let z = array![-0.5_f64, 0.5_f64];
        let n = 1;
        let j = jacobi(n, 1.0, 2.0, z.view());

        assert_array_approx_eq!(j.row(0), array![1.0, 1.0], 1e-14);
        assert_array_approx_eq!(j.row(1), array![-1.75, 0.75], 1e-14);
    }

    #[test]
    fn test_degree_zero_base_case() {
        let z = array![0.33_f64, 0.77_f64];
        let j = jacobi(0, 0.0, 0.0, z.view());
        let dj = jacobi_derivative(0, 0.0, 0.0, z.view());

        assert_eq!(j.shape(), &[1, 2]);
        assert_array_approx_eq!(j.row(0), array![1.0, 1.0], 1e-14);
        assert_eq!(dj.shape(), &[1, 2]);
        assert_array_approx_eq!(dj.row(0), array![0.0, 0.0], 1e-14);
    }

    #[test]
    fn test_vandermonde_inversion() {
        use ndarray_linalg::Inverse;

        let z = array![-1.0_f64, 0.0_f64, 1.0_f64];
        let n = 2;
    
        let j = jacobi(n, 0.0, 0.0, z.view()); 
        let v = j.t().to_owned();
        let v_inv = v.inv().expect("vandermonde matrix inversion failed");
        let identity = v.dot(&v_inv);

        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (identity[[i, j]] - expected).abs() < 1e-13,
                    "identity matrix mismatch at [{}, {}]: got {}, expected {}",
                    i, j, identity[[i, j]], expected
                );
            }
        }
    }
}