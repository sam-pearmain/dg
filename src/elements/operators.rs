use ndarray::ArrayView2;

use crate::{elements::basis::Basis, float::Float};

/// Matrix operators build upon the basis functions
pub trait Operators<F: Float>: Basis<F> {
    /// The projection matrix to interpolate interior solution points to interface flux points
    fn interior_interface_projection(&self) -> ArrayView2<'_, F>;
    /// The weak divergence matrix used to integrate an interior vector field against the gradient of the basis functions
    fn interior_flux_weak_divergence(&self) -> ArrayView2<'_, F>;
    /// The projection matrix to extrapolate an interior vector field to the interface and compute its component parallel to the face normal
    fn interior_interface_normal_projection(&self) -> ArrayView2<'_, F>;
    /// The gradient of an interior scalar field
    fn interior_gradient(&self) -> ArrayView2<'_, F>;
    /// Projects the interior solution onto the quadrature points
    fn interior_quadrature_projection(&self) -> ArrayView2<'_, F>;
    /// The divergence of the correction basis
    fn interface_flux_correction_projection(&self) -> ArrayView2<'_, F>;
    /// The interface
    fn interface_gradient_correction_projection(&self) -> ArrayView2<'_, F>;
    /// The L2 projection matrix that integrates a scalar field at quadrature points and projects the result against the solution points
    fn quadrature_scalar_projection(&self) -> ArrayView2<'_, F>;
    /// The L2 projection matrix that integrates a vector field at quadrature points and projects the result against the solution points
    fn quadrature_vector_projection(&self) -> ArrayView2<'_, F>;
}
