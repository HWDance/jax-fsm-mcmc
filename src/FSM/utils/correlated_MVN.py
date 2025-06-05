import jax.numpy as jnp

def get_logpdf_fn(corr, dim):

    mean = jnp.zeros(dim)
    cov = jnp.array([[corr if i != j else 1.0 for j in range(dim)] for i in range(dim)])
    cov_inv = jnp.linalg.inv(cov)
    log_det_cov = jnp.linalg.slogdet(cov)[1]
    mean1 = jnp.array([10.0] * dim)
    mean2 = jnp.array([-10.0] * dim)
    cov1 = jnp.array([[corr if i != j else 1.0 for j in range(dim)] for i in range(dim)])
    cov2 = jnp.array([[corr if i != j else 1.0 for j in range(dim)] for i in range(dim)])
    
    cov1_inv = jnp.linalg.inv(cov1)
    cov2_inv = jnp.linalg.inv(cov2)
    log_det_cov1 = jnp.linalg.slogdet(cov1)[1]
    log_det_cov2 = jnp.linalg.slogdet(cov2)[1]
        
    def logpdf_fn(x):
    
        log_gaussian = -0.5 * (x - mean).T @ cov_inv @ (x - mean) - 0.5 * log_det_cov
    
        log_mode1 = -0.5 * (x - mean1).T @ cov1_inv @ (x - mean1) - 0.5 * log_det_cov1
        log_mode2 = -0.5 * (x - mean2).T @ cov2_inv @ (x - mean2) - 0.5 * log_det_cov2
    
        log_mixture = jnp.logaddexp(log_mode1, log_mode2)
    
        log_target = log_gaussian + log_mixture
        
        return log_target

    return logpdf_fn

def get_logpdf_fn_base(corr, dim):

    mean = jnp.zeros(dim)
    cov = jnp.array([[corr if i != j else 1.0 for j in range(dim)] for i in range(dim)])
    cov_inv = jnp.linalg.inv(cov)
    log_det_cov = jnp.linalg.slogdet(cov)[1]
        
    def logpdf_fn(x):
    
        log_gaussian = -0.5 * (x - mean).T @ cov_inv @ (x - mean) - 0.5 * log_det_cov
        
        return log_gaussian

    return logpdf_fn
