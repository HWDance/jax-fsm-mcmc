import jax
import jax.numpy as jnp
from jax.scipy.linalg import cho_solve, cholesky

# Logprob definition
def rbf_kernel(X, lengthscale, scale):
    """Compute the RBF (Gaussian) kernel matrix."""
    pairwise_dists = jnp.sum(X**2, axis=1).reshape(-1, 1) + jnp.sum(X**2, axis=1) - 2 * jnp.dot(X, X.T)
    return jnp.exp(scale) * jnp.exp(-0.5 * pairwise_dists / jnp.exp(lengthscale))

def logpdf_gp_fn(y, X):
    """
    Returns a log-probability function with `y` and `X` frozen, 
    so it only takes `params` (vector of [sigma, lengthscale, scale]) as input.
    """
    @jax.jit
    def logpdf_gp(params):
        """
        Calculate the log-probability of observations `y` under a GP with RBF kernel.

        Parameters:
        - params: JAX array of shape (3,), where:
            - params[0] = sigma (noise variance)
            - params[1] = lengthscale (bandwidth)
            - params[2] = scale (signal variance)

        Returns:
        - log_prob: The log-probability of the GP observations
        """
        sigma, lengthscale, scale = params[0], params[1], params[2]

        # Compute the kernel matrix K(X, X)
        K = rbf_kernel(X, lengthscale, scale)
        K += jnp.exp(sigma) * jnp.eye(X.shape[0])  # Add noise term to the diagonal

        # Compute the Cholesky decomposition of K
        L = cholesky(K, lower=True)

        # Solve for alpha = K^(-1) y using the Cholesky decomposition
        alpha = cho_solve((L, True), y)
        
        # Log probability
        log_prob = -0.5 * jnp.dot(y, alpha)
        log_prob -= jnp.sum(jnp.log(jnp.diag(L)))  # Log determinant term

        # Prior
        log_prob -= 0.5*sigma**2 + 0.5*lengthscale**2 + 0.5*scale**2
        return log_prob

    return logpdf_gp

    
