import jax
import jax.numpy as jnp
import jax.random as jrnd
from jax.scipy.stats import norm

#########################################
# Data Definition (from JSON)
#########################################

data = {
    "N": 40,
    "n_groups": 5,
    "n_scenarios": 8,
    "scenario_id": jnp.array([1, 2, 3, 4, 5, 6, 7, 8,
                              1, 2, 3, 4, 5, 6, 7, 8,
                              1, 2, 3, 4, 5, 6, 7, 8,
                              1, 2, 3, 4, 5, 6, 7, 8,
                              1, 2, 3, 4, 5, 6, 7, 8]),
    "group_id": jnp.array([1, 1, 1, 1, 1, 1, 1, 1,
                           2, 2, 2, 2, 2, 2, 2, 2,
                           3, 3, 3, 3, 3, 3, 3, 3,
                           4, 4, 4, 4, 4, 4, 4, 4,
                           5, 5, 5, 5, 5, 5, 5, 5]),
    "y": jnp.array([0.375, 0, 0.375, 0, 0.333333333333333, 1, 0.125, 1,
                    0.25, 0, 0.5, 0.125, 0.5, 1, 0.125, 0.857142857142857,
                    0.5, 0.666666666666667, 0.333333333333333, 0, 0.142857142857143, 1, 0, 1,
                    0.142857142857143, 0, 0.714285714285714, 0, 0.285714285714286, 1,
                    0.142857142857143, 1, 0.428571428571429, 0, 0.285714285714286,
                    0.857142857142857, 0.857142857142857, 0.857142857142857, 0.142857142857143, 0.75])
}

# Adjust indices from Stan (1-indexed) to Python (0-indexed)
data["group_id"] = data["group_id"] - 1
data["scenario_id"] = data["scenario_id"] - 1

#########################################
# Model LogPDF Function
#########################################

def model_logpdf(params, data):
    """
    Compute the log joint density for the hierarchical group–scenario model.
    
    Data:
      - N: number of observations.
      - n_groups: number of groups.
      - n_scenarios: number of scenarios.
      - group_id: integer array of length N (each entry in {0,..., n_groups-1})
      - scenario_id: integer array of length N (each entry in {0,..., n_scenarios-1})
      - y: response vector of length N.
      
    Parameters:
      params is a dict with keys:
        - "a": vector of length n_groups.
        - "b": vector of length n_scenarios.
        - "mu_a": scalar.
        - "mu_b": scalar.
        - "sigma_a": scalar (positive, assumed uniform (0,100)).
        - "sigma_b": scalar (positive, assumed uniform (0,100)).
        - "sigma_y": scalar (positive, assumed uniform (0,100)).
    
    Model:
      For each observation i:
         y_hat[i] = a[group_id[i]] + b[scenario_id[i]]
         y[i] ~ Normal(y_hat[i], sigma_y)
      
      Priors:
         mu_a ~ Normal(0,1)
         a ~ Normal(10 * mu_a, sigma_a)
         mu_b ~ Normal(0,1)
         b ~ Normal(10 * mu_b, sigma_b)
         (sigma_a, sigma_b, sigma_y) have implicit uniform(0,100) priors.
    
    Returns:
      A scalar: the log joint density.
    """
    N = data["N"]
    group_id = data["group_id"]
    scenario_id = data["scenario_id"]
    y = data["y"]
    
    a = params["a"]             # shape (n_groups,)
    b = params["b"]             # shape (n_scenarios,)
    mu_a = params["mu_a"]       # scalar
    mu_b = params["mu_b"]       # scalar
    sigma_a = params["sigma_a"] # scalar
    sigma_b = params["sigma_b"] # scalar
    sigma_y = params["sigma_y"] # scalar
    
    # Compute linear predictor for each observation.
    # Using advanced indexing: for each i, pick a[group_id[i]] and b[scenario_id[i]]
    y_hat = a[group_id] + b[scenario_id]  # shape (N,)
    
    # Log-likelihood for observations.
    logp_obs = jnp.sum(norm.logpdf(y, loc=y_hat, scale=sigma_y))
    
    # Priors:
    logp_mu_a = norm.logpdf(mu_a, loc=0, scale=1)
    logp_a = jnp.sum(norm.logpdf(a, loc=10 * mu_a, scale=sigma_a))
    
    logp_mu_b = norm.logpdf(mu_b, loc=0, scale=1)
    logp_b = jnp.sum(norm.logpdf(b, loc=10 * mu_b, scale=sigma_b))
    
    # Uniform priors on sigma parameters are constant over the allowed range (0,100),
    # so we omit them (or you can add 0 if you prefer).
    
    return logp_obs + logp_mu_a + logp_a + logp_mu_b + logp_b

def logpdf_fn(params):
    return model_logpdf(params, data)

#########################################
# Parameter Initialization
#########################################

def initialize_parameters(rng):
    """
    Stochastically initialize parameters for the hierarchical model.
    
    Args:
      rng: A JAX random key.
      
    Returns:
      A dict with keys:
        - "a": vector of length n_groups, initialized from Normal(0,1).
        - "b": vector of length n_scenarios, initialized from Normal(0,1).
        - "mu_a": scalar from Normal(0,1).
        - "mu_b": scalar from Normal(0,1).
        - "sigma_a": positive scalar from a half-normal (e.g., abs(normal()) + small constant).
        - "sigma_b": positive scalar similarly.
        - "sigma_y": positive scalar similarly.
    """
    n_groups = data["n_groups"]
    n_scenarios = data["n_scenarios"]
    
    rng, key_a, key_b, key_mu_a, key_mu_b, key_sigma_a, key_sigma_b, key_sigma_y = jrnd.split(rng, 8)
    
    a = jrnd.normal(key_a, (n_groups,))
    b = jrnd.normal(key_b, (n_scenarios,))
    mu_a = jrnd.normal(key_mu_a, ())
    mu_b = jrnd.normal(key_mu_b, ())
    sigma_a = jnp.abs(jrnd.normal(key_sigma_a, ())) + 0.1
    sigma_b = jnp.abs(jrnd.normal(key_sigma_b, ())) + 0.1
    sigma_y = jnp.abs(jrnd.normal(key_sigma_y, ())) + 0.1
    
    return {
        "a": a,
        "b": b,
        "mu_a": mu_a,
        "mu_b": mu_b,
        "sigma_a": sigma_a,
        "sigma_b": sigma_b,
        "sigma_y": sigma_y
    }

#########################################
# HierarchicalModel Class
#########################################

class HierarchicalModel:
    def __init__(self, data):
        self.data = data
    
    def logprob_fn(self, params):
        return model_logpdf(params, self.data)
    
    def initialize_model(self, rng, num_chains):
        chain_rng = jrnd.split(rng, num_chains)
        self.init_params = jax.vmap(lambda key: initialize_parameters(key))(chain_rng)
        return self.init_params

#########################################
# __main__ Test
#########################################

if __name__ == "__main__":
    # Instantiate the model with the given data.
    dist = HierarchicalModel(data)
    rng = jrnd.PRNGKey(0)
    num_chains = 2
    dist.initialize_model(rng, num_chains)
    
    # Evaluate the log joint density for each chain.
    logp_values = jax.vmap(dist.logprob_fn)(dist.init_params)
    print("Log Likelihood (random inits):", logp_values)
