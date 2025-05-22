import jax 
import jax.numpy as jnp
import jax.lax as lax
from jax import Array
from jax.typing import ArrayLike
from ...utils import _jax_init

_jax_init()

def _legendre_poly_and_deriv_scalar(n: int, x: ArrayLike) -> tuple[Array, Array]:
    x = jnp.asarray(x)

    p0 = jnp.array(1.0)
    dp0 = jnp.array(0.0)

    if n == 0:
        return p0, dp0

    p1 = x
    dp1 = jnp.array(1.0)
    
    if n == 1:
        return p1, dp1
    
    def body(i: int, state: tuple[Array, Array, Array, Array]) -> tuple[Array, Array, Array, Array]:
        pn_minus_1, pn_minus_2, dpn_minus_1, dpn_minus_2 = state

        pn: Array = (
            (2.0 * i - 1.0) * x * pn_minus_1 - (i - 1.0) * pn_minus_2 
        ) / i
        dpn: Array = dpn_minus_2 + (2.0 * i - 1.0) * pn_minus_1

        return pn, pn_minus_1, dpn, dpn_minus_1 
    
    initial_state = (p1, p0, dp1, dp0)
    final_state = lax.fori_loop(2, n + 1, body, initial_state)
    pn = final_state[0]
    dpn = final_state[2]

    return pn, dpn

def _newton_raphson_roots(n: int, intitial_guess: jax.Array, max_iter: int = 50, tol: float = 1e-15):
    eval_pn_dpn_vec = jax.vmap(lambda x: _legendre_poly_and_deriv_scalar(n, x), in_axes = 0)

    def cond(state):
        iter_idx, _, current_max_abs_update = state
        return (iter_idx < max_iter) & (current_max_abs_update > tol)
    
    def body(state):
        iter_idx, roots_current_iter, _ = state

        pn_vals, dpn_vals = eval_pn_dpn_vec(roots_current_iter)
        delta_x = pn_vals / (dpn_vals + 1e-30)

        roots_new_iter = roots_current_iter - delta_x
        current_max_abs_update = jnp.max(jnp.abs(delta_x))

        return iter_idx + 1, roots_new_iter, current_max_abs_update 

    initial_state = (0, intitial_guess, jnp.array(2.0 * tol, dtype = intitial_guess.dtype))
    _, final_roots, _ = lax.while_loop(cond, body, initial_state)
    return final_roots

def gauss_legendre_points_weights(n_points: int):
    if n_points <= 0:
        raise ValueError("number of points, n, must be a positive integer")
    
    if n_points == 1: 
        return jnp.array(0.0), jnp.array(2.0)
    
    n_vals = jnp.arange(1, n_points + 1)
    initial_guesses = jnp.cos(jnp.pi * (n_vals - 0.25) / (n_points + 0.5))
    initial_guesses = jnp.asarray(initial_guesses)

    points = _newton_raphson_roots(n_points, initial_guesses)
    _, dnp_at_points = jax.vmap(lambda r: _legendre_poly_and_deriv_scalar(n_points, r))(points)
    weights = 2.0 / ((1.0 - points**2) * dnp_at_points**2)

    sorted_indices = jnp.argsort(points)
    points = points[sorted_indices]
    weights = weights[sorted_indices]

    return points, weights

GAUSS_LEGENDRE_LINE_QUADRATURE_POINTS = {
    n: gauss_legendre_points_weights(n)[0] for n in range(1, 20)
}

GAUSS_LEGENDRE_LINE_QUADRATURE_WEIGHTS = {
    n: gauss_legendre_points_weights(n)[1] for n in range(1, 20)
}

if __name__ == "__main__":
    for n, points in GAUSS_LEGENDRE_LINE_QUADRATURE_POINTS.items():
        weights = GAUSS_LEGENDRE_LINE_QUADRATURE_WEIGHTS[n]
        print(n, points, weights)