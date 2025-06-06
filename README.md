# JAX-FSM-MCMC: Vectorized MCMC without synchronization barriers

This repository contains JAX implementations of several stochastic-length proposal MCMC algorithms (i.e. HMC-NUTS and Slice sampling variants) for more efficient execution on SIMD architectures, when vectorizating with `vmap`. The implementation method is based on the paper "Efficiently Vectorized MCMC on Modern Accelerators" [https://www.arxiv.org/abs/2503.17405](https://www.arxiv.org/abs/2503.17405). 

### TL;DR

- **Issue:** Wrapping a data-dependent `while` loop in `jax.vmap` produces a single batched `While` operation with an aggregated termination condition for the whole batch, so each iteration stalls until **all** batch elements finish—creating a full-batch synchronization barrier.

- **Solution:** We split the computation at each loop boundary into separate blocks, then use `jax.lax.switch` or `jax.lax.cond` to dispatch each block until every chain recovers its required samples. This lets each Markov chain progress through its own state sequence independently, eliminating the synchronization barrier.

## Table of Contents

1. [Installation](#installation)  
2. [Getting Started](#getting-started)  
   - [Basic Usage](#basic-usage)  
   - [API Overview](#api-overview)  
3. [Reproducing Experiments](#reproducing-experiments)  
4. [License](#license) 

## Installation

### 1. Prerequisites
requirements.txt

### 2. Clone and Install this repo

```bash
git clone https://github.com/hwdance/jax-fsm-mcmc.git
cd jax-fsm-mcmc
pip install -e .
```

### 3. Test Installation
```python
import jax_fsm_mcmc
print(jax_fsm_mcmc.__version__)
```

## Getting Started 
Below is a minimal example showing how to run a collection of NUTS chains using the FSM approach. For more advanced usage (slice samplers, customization, etc.), see the API Overview section.


### Basic Usage 
 Import jax, numpy and the FSM machinery we will use to sample MCMC chains with NUTS.
 
```python
# Basic imports
import jax
import jax.numpy as jnp
import jax.random as jrnd
import numpy as np

# FSM imports
from FSM.mcmc.nuts_bundle import NutsFSM # NUTS algorithm in FSM form
from FSM.base.run_fsm_in import jitted_update # Jitted function to call blocks of the FSM
from FSM.base.run_fsm_in import get_fsm_samples_chain as get_fsm_samples # Outer wrapper to get n-samples per chain
```

We will implement nuts to learn the covariance hyperparameter posterior of a Gaussian process $f$, given data $(X_i,Y_i) \sim_{iid} P$ where $Y = f(X) + \xi : \xi \sim N(0,1)$. We start off by defining the log-posterior of the covariance hyperparameters $\tau,\eta, \sigma$. The likelihood is $p(y,X|\tau,\eta,\sigma) = N(y|0,K_{XX}(\tau,\eta) + I\sigma^2)$, where $[K_{XX}]_{i,j} = \tau \exp(-\frac{1}{2\eta^2}\|x_i-x_j\|^2)$, and we use standard Gaussian priors $\sigma,\eta,\tau \sim N(0,1)$.

```python
# Helper to create log likelihood for GPR
from FSM.utils.gpr import logpdf_gp_fn as get_logpdf_fn

# Generate data using linear model
def generate_linear_XY(key, n, x_min=-3.0, x_max=3.0):
    """
    Generate n pairs (X, Y) where:
      X are linearly spaced between x_min and x_max,
      U ~ Normal(0, 1),
      Y = X + U
    """
    X = jnp.linspace(x_min, x_max, n)
    key, subkey = jax.random.split(key)
    U = jax.random.normal(subkey, shape=(n,))
    Y = X + U
    
    return X, Y, key

key = jax.random.PRNGKey(42)
n_samples = 500

X, y, key = generate_linear_XY(key, n_samples, x_min=-3.0, x_max=3.0)

print("X shape:", X.shape)
print("Y shape:", y.shape)

# Define log-posterior of covariance hyperparameters
logprob_fn = jax.jit(get_logpdf_fn(y, X))

```




### API Overview


## Reproducing Experiments

## License


