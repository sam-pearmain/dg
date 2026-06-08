use ndarray::Array2;

use crate::{basis::Basis, float::Float};

/// Matrix operators build upon the basis functions
pub trait Operators<F: Float>: Basis<F> {
    /// The projection matrix to interpolate interior solution points to interface flux points
    fn interior_interface_projection(&self) -> Array2<F>;
    /// The weak divergence matrix used to integrate an interior vector field against the gradient of the basis functions
    fn interior_flux_weak_divergence(&self) -> Array2<F>;
    /// The projection matrix to extrapolate an interior vector field to the interface and compute its component parallel to the face normal
    fn interior_interface_normal_projection(&self) -> Array2<F>;
    /// The gradient of an interior scalar field
    fn interior_gradient(&self) -> Array2<F>;
    /// Projects the interior solution onto the quadrature points
    fn interior_quadrature_projection(&self) -> Array2<F>;
    /// The divergence of the correction basis
    fn interface_flux_correction_projection(&self) -> Array2<F>;
    /// The interface
    fn interface_gradient_correction_projection(&self) -> Array2<F>;
    /// The L2 projection matrix that integrates a scalar field at quadrature points and projects the result against the solution points
    fn quadrature_scalar_projection(&self) -> Array2<F>;
    /// The L2 projection matrix that integrates a vector field at quadrature points and projects the result against the solution points
    fn quadrature_vector_projection(&self) -> Array2<F>;
}
