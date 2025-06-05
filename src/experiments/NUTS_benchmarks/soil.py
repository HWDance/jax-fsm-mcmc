import jax
import jax.numpy as jnp
import jax.random as jrnd
from jax.scipy.stats import norm, beta, cauchy
import jax.scipy.special as jsp_special
import diffrax

#############################################
# ODE Function: Two-Pool Feedback System
#############################################

def two_pool_feedback(t, C, theta, x_r, x_i):
    """
    ODE system for the two-pool model with feedback.
    
    Args:
      t: time (real)
      C: state vector, shape (2,). C[0] = pool1, C[1] = pool2.
      theta: parameter vector of length 4, with
             theta[0] = k1,
             theta[1] = k2,
             theta[2] = alpha21,
             theta[3] = alpha12.
      x_r, x_i: extra data (unused)
    
    Returns:
      dC_dt: derivative vector, shape (2,)
    """
    k1 = theta[0]
    k2 = theta[1]
    alpha21 = theta[2]
    alpha12 = theta[3]
    
    dC_dt0 = -k1 * C[0] + alpha12 * k2 * C[1]
    dC_dt1 = -k2 * C[1] + alpha21 * k1 * C[0]
    return jnp.array([dC_dt0, dC_dt1])

#############################################
# Simple Euler Integrator (for demonstration)
#############################################

def integrate_ode(ode_fn, C0, t0, ts, theta, x_r, x_i):
    """
    Simple Euler integrator over times ts.
    This is only for demonstration; in practice, one would use a higher-order method.
    
    Args:
      ode_fn: function(t, C, theta, x_r, x_i) -> dC_dt.
      C0: initial state, shape (2,).
      t0: initial time.
      ts: array of measurement times, shape (N_t,).
      theta: parameter vector, shape (4,).
      x_r, x_i: extra data (unused here).
      
    Returns:
      C_hat: array of shape (N_t, 2) with the state at each measurement time.
    """
    N_t = ts.shape[0]
    def step_fn(c, t_next):
        current_time, current_state = c
        dt = t_next - current_time
        dstate = ode_fn(current_time, current_state, theta, x_r, x_i)
        new_state = current_state + dt * dstate
        return (t_next, new_state), new_state
    # Use lax.scan over ts.
    (final_time, final_state), states = jax.lax.scan(step_fn, (t0, C0), ts)
    return states


def integrate_ode_diffrax(ode_fn, C0, t0, ts, theta, x_r, x_i):
    """
    Integrate an ODE using diffrax's Euler solver.

    Args:
      ode_fn: function(t, y, theta, x_r, x_i) -> dy/dt.
      C0: initial state, shape (2,).
      t0: initial time.
      ts: 1D array of measurement times (monotonic), shape (N_t,).
      theta: parameter vector, shape (4,).
      x_r, x_i: extra data (unused here).

    Returns:
      An array of shape (N_t, 2) containing the state at each measurement time.
    """
    # Use diffrax's Euler method with a fixed dt0.
    solver = diffrax.Euler()
    # Set dt0 based on the spacing between measurement times.
    dt0 = ts[1] - ts[0]
    # Use diffrax's SaveAt to specify that we want the solution at times ts.
    saveat = diffrax.SaveAt(ts=ts)
    
    sol = diffrax.diffeqsolve(
        diffrax.ODETerm(lambda t, y, args: ode_fn(t, y, *args)),
        solver,
        t0=t0,
        t1=ts[-1],
        dt0=dt0,
        y0=C0,
        saveat=saveat,
        args=(theta, x_r, x_i)
    )
    return sol.ys
    
#############################################
# Evolved CO2 Function
#############################################

def evolved_CO2(N_t, t0, ts, gamma, totalC_t0, k1, k2, alpha21, alpha12, x_r, x_i):
    """
    Compute the evolved CO2 over time.
    
    Args:
      N_t: number of measurement times.
      t0: initial time.
      ts: measurement times, shape (N_t,).
      gamma: partitioning coefficient (between 0 and 1).
      totalC_t0: initial total carbon.
      k1, k2, alpha21, alpha12: ODE parameters.
      x_r, x_i: extra data (empty arrays).
      
    Returns:
      eCO2_hat: vector of length N_t; for each time, totalC_t0 minus current pool sum.
    """
    # Initial state: split total carbon between pools.
    C_t0 = jnp.array([gamma * totalC_t0, (1 - gamma) * totalC_t0])
    # Parameter vector theta:
    theta = jnp.array([k1, k2, alpha21, alpha12])
    # Call the integrator to simulate the ODE.
    C_hat = integrate_ode(two_pool_feedback, C_t0, t0, ts, theta, x_r, x_i)
    # For each time t, evolved CO2 is totalC_t0 minus sum of pools.
    def compute_eCO2(C_state):
        return totalC_t0 - jnp.sum(C_state)
    eCO2_hat = jax.vmap(compute_eCO2)(C_hat)
    return eCO2_hat

#############################################
# Epidemic Two-Pool Model LogPDF
#############################################

def ep_model_logpdf(params, data):
    """
    Compute the log joint density for the two-pool carbon model.
    
    Data is a dict with keys:
      - totalC_t0: initial total carbon (real, >0)
      - t0: initial time (real)
      - N_t: number of measurement times (integer)
      - ts: measurement times (array of length N_t)
      - eCO2mean: observed cumulative evolved CO2 (array of length N_t)
      (x_r and x_i are empty)
    
    Free parameters:
      - k1, k2: positive real numbers (decomposition rates)
      - alpha21, alpha12: positive real numbers (transfer coefficients)
      - gamma: partitioning coefficient (real in [0,1])
      - sigma: observation standard deviation (positive)
    
    Priors:
      - gamma ~ Beta(10, 1)
      - k1 ~ Normal(0,1)
      - k2 ~ Normal(0,1)
      - alpha21 ~ Normal(0,1)
      - alpha12 ~ Normal(0,1)
      - sigma ~ Cauchy(0,1)
    
    Likelihood:
      For t=1,...,N_t, eCO2mean[t] ~ Normal(eCO2_hat[t], sigma),
      where eCO2_hat is computed by evolving the ODE.
    """
    totalC_t0 = data["totalC_t0"]
    t0 = data["t0"]
    N_t = data["N_t"]
    ts = data["ts"]
    eCO2mean = data["eCO2mean"]
    # x_r and x_i are empty arrays.
    x_r = jnp.array([])
    x_i = jnp.array([], dtype=jnp.int32)
    
    # Extract free parameters.
    k1 = params["k1"]
    k2 = params["k2"]
    alpha21 = params["alpha21"]
    alpha12 = params["alpha12"]
    gamma = params["gamma"]
    sigma = params["sigma"]
    
    # Compute evolved CO2 predictions.
    eCO2_hat = evolved_CO2(N_t, t0, ts, gamma, totalC_t0, k1, k2, alpha21, alpha12, x_r, x_i)
    
    logp = 0.0
    # Likelihood for observed evolved CO2.
    for t in range(N_t):
        logp += norm.logpdf(eCO2mean[t], loc=eCO2_hat[t], scale=sigma)
    
    # Priors:
    logp += beta.logpdf(gamma, 10, 1)
    logp += norm.logpdf(k1, 0, 1)
    logp += norm.logpdf(k2, 0, 1)
    logp += norm.logpdf(alpha21, 0, 1)
    logp += norm.logpdf(alpha12, 0, 1)
    logp += cauchy.logpdf(sigma, 0, 1)
    
    return logp

def logpdf_fn(params):
    return ep_model_logpdf(params, data)

#############################################
# Parameter Initialization
#############################################

def initialize_ep_parameters(rng):
    """
    Stochastically initialize parameters for the two-pool carbon model.
    
    Returns a dict with keys:
      - k1, k2, alpha21, alpha12: drawn from Normal(0,1) and then made positive (absolute value)
      - gamma: drawn from a Uniform(0,1) (or Beta(1,1))
      - sigma: drawn from abs(normal()) plus a small constant.
    """
    rng, key_k1, key_k2, key_a21, key_a12, key_gamma, key_sigma = jrnd.split(rng, 7)
    
    k1 = jnp.abs(jrnd.normal(key_k1, ()))  # make positive
    k2 = jnp.abs(jrnd.normal(key_k2, ()))
    alpha21 = jnp.abs(jrnd.normal(key_a21, ()))
    alpha12 = jnp.abs(jrnd.normal(key_a12, ()))
    gamma = jrnd.beta(key_gamma, 1.0, 1.0)  # Uniform(0,1) equivalent
    sigma = jnp.abs(jrnd.normal(key_sigma, ())) + 0.1
    
    return {
        "k1": k1,
        "k2": k2,
        "alpha21": alpha21,
        "alpha12": alpha12,
        "gamma": gamma,
        "sigma": sigma
    }

#############################################
# TwoPoolModel Class
#############################################

class TwoPoolModel:
    def __init__(self, data):
        self.data = data
    
    def logprob_fn(self, params):
        return ep_model_logpdf(params, self.data)
    
    def initialize_model(self, rng, num_chains):
        chain_rng = jrnd.split(rng, num_chains)
        self.init_params = jax.vmap(lambda key: initialize_ep_parameters(key))(chain_rng)
        return self.init_params

#############################################
# Data for the Two-Pool Model (from JSON)
#############################################

data = {
    "totalC_t0": 7.7,
    "t0": 0.0,
    "N_t": 25,
    "ts": jnp.array([1.15601851851852, 2.03449074074074, 4.96157407407407, 6.01712962962963,
                     7.03796296296296, 8.01365740740741, 9.02060185185185, 12.0219907407407,
                     13.0574074074074, 13.9740740740741, 14.9525462962963, 15.9525462962963,
                     19.0032407407407, 19.9685185185185, 20.9518518518519, 21.9650462962963,
                     22.9219907407407, 26.012962962963, 27.050462962963, 27.9310185185185,
                     28.9386574074074, 29.9386574074074, 39.9636574074074, 40.9372685185185,
                     41.9726851851852]),
    "eCO2mean": jnp.array([0.0277780268338887, 0.719282564617987, 2.06157264777962, 1.83912852534761,
                           2.79084379741546, 2.67025926226335, 3.25393027352365, 3.87829730899174,
                           4.16129834125508, 4.83147131226128, 4.61616507571551, 5.13749850718396,
                           5.95377256441306, 5.91393741365973, 5.94829695445491, 5.81102598536745,
                           5.75313320451491, 6.02526731239607, 6.68894423430123, 6.57366504262456,
                           6.27822369172602, 6.80503239142778, 8.22979270757457, 7.75007806202788,
                           7.63721799990632])
}

#############################################
# __main__ Test
#############################################

if __name__ == "__main__":
    model = TwoPoolModel(data)
    rng = jrnd.PRNGKey(0)
    num_chains = 2
    model.initialize_model(rng, num_chains)
    logp_values = jax.vmap(model.logprob_fn)(model.init_params)
    print("Log Likelihood (random inits):", logp_values)
