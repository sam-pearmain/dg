use ndarray::{Array1, ArrayView1};

use crate::float::Float;

/// Computes  the Jacobi polynomial $P_n^{(\alpha, \beta)(\mathbf{x})}$.
/// These polynomials are orthogonal on [-1, 1] with weight function $w(x) = (1 - x)^{\alpha} (1 + x)^{\beta}$
pub fn jacobi<F: Float>(n: usize, alpha: F, beta: F, x: ArrayView1<'_, F>) -> Array1<F> {
    let one = F::one();
    let two = F::from(2).unwrap();

    let mut j = Array1::<F>::ones(x.len());

    match n {
        0 => {}
        1 => j.zip_mut_with(&x, |y, x| {
            *y = (alpha - beta + (alpha + beta + two) * *x) / two
        }),
        _ => {
            let mut coeffs = Vec::with_capacity(n - 1);
            for deg in 2..=n {
                let deg_as_float = F::from(deg).unwrap();
                let a1 = two
                    * deg_as_float
                    * (deg_as_float + alpha + beta)
                    * (two * deg_as_float + alpha + beta - two);
                let a2 = (two * deg_as_float + alpha + beta - one) * (alpha * alpha - beta * beta);
                let a3 = (two * deg_as_float + alpha + beta - one)
                    * (two * deg_as_float + alpha + beta)
                    * (two * deg_as_float + alpha + beta - two);
                let a4 = two
                    * (deg_as_float + alpha - one)
                    * (deg_as_float + beta - one)
                    * (two * deg_as_float + alpha + beta);

                coeffs.push((a1, a2, a3, a4));
            }

            j.zip_mut_with(&x, |y, val| {
                let mut p0 = one;
                let mut p1 = (alpha - beta + (alpha + beta + two) * *val) / two;

                for &(a1, a2, a3, a4) in &coeffs {
                    let p2 = ((a2 + a3 * *val) * p1 - a4 * p0) / a1;
                    p0 = p1;
                    p1 = p2;
                }
                *y = p1;
            });
        }
    }

    j
}

/// Computes the derivative of the Jacobi polynomial $\frac{d}{dx} P_n^{(\alpha, \beta)(\mathbf{x})}$
pub fn jacobi_derivative<F: Float>(n: usize, alpha: F, beta: F, x: ArrayView1<'_, F>) -> Array1<F> {
    let one = F::one();
    let two = F::from(2).unwrap();

    let mut dj = Array1::zeros(x.len());

    match n {
        0 => {}
        _ => {
            let deg_as_float = F::from(n).unwrap();
            dj.assign(
                &jacobi(n - 1, alpha + one, beta + one, x)
                    .map(|dy| *dy * (deg_as_float + alpha + beta + one) / two),
            );
        }
    };

    dj
}

/// Computes the Legendre polynomial $P_n(x)$.
/// These polynomials are orthogonal on the interval [-1, 1] with weight functions $w(x) = 1$
pub fn legendre<F: Float>(degree: usize, points: ArrayView1<'_, F>) -> Array1<F> {
    jacobi(degree, F::zero(), F::zero(), points)
}

/// Computes the derivative of the Legendre polynomial $\frac{d}{dx} P_n(x)$
pub fn legendre_derivative<F: Float>(degree: usize, points: ArrayView1<'_, F>) -> Array1<F> {
    jacobi_derivative(degree, F::zero(), F::zero(), points)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_legendre_polynomials() {
        let x = array![-1.0, -0.5, 0.0, 0.5, 1.0];

        let p0 = legendre(0, x.view());
        let expected_p0 = array![1.0, 1.0, 1.0, 1.0, 1.0];
        assert_eq!(&p0, &expected_p0);

        let p1 = legendre(1, x.view());
        let expected_p1 = array![-1.0, -0.5, 0.0, 0.5, 1.0];
        assert_eq!(&p1, &expected_p1);

        let p2 = legendre(2, x.view());
        let expected_p2 = array![1.0, -0.125, -0.5, -0.125, 1.0];
        assert_eq!(&p2, &expected_p2);
    }

    #[test]
    fn test_legendre_derivatives() {
        let x = array![-1.0, -0.5, 0.0, 0.5, 1.0];

        let dp0 = legendre_derivative(0, x.view());
        let expected_dp0 = array![0.0, 0.0, 0.0, 0.0, 0.0];
        assert_eq!(&dp0, &expected_dp0);

        let dp1 = legendre_derivative(1, x.view());
        let expected_dp1 = array![1.0, 1.0, 1.0, 1.0, 1.0];
        assert_eq!(&dp1, &expected_dp1);

        let dp2 = legendre_derivative(2, x.view());
        let expected_dp2 = array![-3.0, -1.5, 0.0, 1.5, 3.0];
        assert_eq!(&dp2, &expected_dp2);
    }

    #[test]
    fn test_jacobi_polynomials() {
        let x = array![-1.0, -0.5, 0.0, 0.5, 1.0];
        let alpha = 1.0;
        let beta = 2.0;

        let p1 = jacobi(1, alpha, beta, x.view());
        let expected_p1 = array![-3.0, -1.75, -0.5, 0.75, 2.0];
        assert_eq!(&p1, &expected_p1);
        
        let dp1 = jacobi_derivative(1, alpha, beta, x.view());
        let expected_dp1 = array![2.5, 2.5, 2.5, 2.5, 2.5];
        assert_eq!(&dp1, &expected_dp1);
    }

    #[test]
    fn test_jacobi_endpoints() {
        let x = array![-1.0, 1.0];
        let alpha = 1.0;
        let beta = 2.0;

        let p2 = jacobi(2, alpha, beta, x.view());
        let expected_p2 = array![6.0, 3.0];
        assert_eq!(&p2, &expected_p2);
    }
}