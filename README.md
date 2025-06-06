# jax-fsm-mcmc: Vectorized MCMC without synchronization barriers

This repository contains JAX implementations of several stochastic-length proposal MCMC algorithms (i.e. HMC-NUTS and Slice sampling variants) for more efficient execution on SIMD architectures, when vectorizating with `vmap`. The implementation method is based on the paper "Efficiently Vectorized MCMC on Modern Accelerators" [https://www.arxiv.org/abs/2503.17405](https://www.arxiv.org/abs/2503.17405), which is accepted as a Spotlight at ICML 2025. 

## Table of Contents

1. [Why Use fsm-mcmc?](#tldr)
2. [Installation](#installation)  
3. [Getting Started](#getting-started)  
4. [Reproducing Experiments](#reproducing-experiments)  
5. [License](#license)
6. [Citation](#citation)
7. [Contact](#contact)

## Why Use FSM-MCMC?

**Issue:** Wrapping a data-dependent `while` loop in `jax.vmap` produces a single batched `While` operation with an aggregated termination condition for the whole batch, so each iteration stalls until **all** batch elements finish. For MCMC algorithms with such data-dependent while loops (e.g. NUTS, slice samplers), this creates a full-batch synchronization barrier at every sampling step, leading to inefficient vectorized execution.

  **Example:** LHS: The distribution of the number of integration steps needed by the No-U-Turn sampler (NUTS) [Hoffman and Gelman (2014)](https://www.jmlr.org/papers/volume15/hoffman14a/hoffman14a.pdf) on a high-dimensional correlated Gaussian Mixture. RHS: The distribution of the *maximum* number of integration steps needed across 500 chains to draw each sample (i.e. the distribution of the \# steps required when running 500 chains with `vmap`). The probability that a chain takes many $(>1000)$ steps is extremely small, but the probability that *at least **one*** chain needs $(>1000)$ steps is nearly one. When using `vmap` to run the chains, they will therefore all have to wait for $>1000$ loop iterations to draw each sample.

  ![\# Integration steps taken by HMC NUTS on a correlated Gaussian](HMC_synchprob_.png)

  <br />

**Solution:** 
- We split the computation at each loop boundary into separate blocks $S_1,...,S_K$ which transition to one another based on the while loop terminators. These blocks and transition rules define a *finite state machine* (FSM) - see LHS figure below.
- We use these blocks to define a `step` function which, given a current state $k$ and input variables $z = (x,\log p(x),...)$ to all blocks, (i) checks the current MCMC algorithm block and (ii) uses `jax.lax.switch` or `jax.lax.cond` to dispatch the relevant block to update $z$.
- Starting from initialization $(z_0,k=0)$, we use an outer wrapper to iteratively call `step` until the chain recovers its required samples.
- For vectorization, we just call `vmap(step)` instead of `step`, until all chains have collected their samples.
- `vmap(step)` lets each Markov chain progress through its own set of state sequences independently, eliminating the synchronization barrier.

 **Back to The Example:** On the RHS figure we show results when implementating NUTS using our FSM procedure on the high-dimensional correlated Gaussian. When $m=100$ chains are used with `vmap`, our procedure leads to speed-ups of ~10x.

   ![FSM_NUTS_FSM_Results](FSM_LHS_NUTS_RHS.png)

 



## Installation

### 1. Prerequisites

Before you begin, make sure you have:

#### Conda (Miniconda/Anaconda)
- Tested with Conda ≥ 4.10.
- If you don’t already have it, install from [Miniconda](https://docs.conda.io/en/latest/miniconda.html).

#### CUDA Toolkit & GPU Drivers (optional)
- If you plan to run on GPU, this repo is tested with **CUDA 12.6** and **cuDNN 8.9** (for JAX 0.4.26).
- Verify that your machine has:
  - NVIDIA GPU with CUDA 12 support
  - NVIDIA driver ≥ 535.86
- If you do _not_ have a compatible GPU, the environment will fall back to CPU-only JAXLIB.

### 2. Clone Repo

```bash
git clone https://github.com/hwdance/jax-fsm-mcmc.git
cd jax-fsm-mcmc
```

### 3. Create + Activate Conda Environment
```bash
conda env create -f environment.yml
conda activate fsm-mcmc
```


### 4. Verify Installation
```python
import jax, jaxlib
import jax_fsm_mcmc
import numpyro, blackjax

print("JAX:", jax.__version__)
print("JAXLIB:", jaxlib.__version__)
print("FSM‐MCMC version:", jax_fsm_mcmc.__version__)
print("NumPyro:", numpyro.__version__)
print("Blackjax:", blackjax.__version__)
```

## Getting Started 
Below is a minimal example showing how to run NUTS using our FSM implementation with JAX's `vmap`.


### Basic Usage 
 Imports for sampling MCMC chains with NUTS via our FSM.
 
```python
# Basic imports
import jax
import jax.numpy as jnp
import jax.random as jrnd
import numpy as np
import blackjax
from time import time

# FSM imports
from FSM.mcmc.nuts_bundle import NutsFSM # NUTS algorithm in FSM form
from FSM.base.run_fsm_in import jitted_update # Jitted function to call blocks of the FSM
from FSM.base.run_fsm_in import get_fsm_samples_chain as get_fsm_samples # Outer wrapper to get n-samples per chain
```

We will implement NUTS to sample from the covariance hyperparameter posterior of a Gaussian process $f$, given data $(X_i,Y_i) \sim_{iid} P$ where $Y = f(X) + \xi : \xi \sim N(0,1)$. We start off by defining the log-posterior of the covariance hyperparameters $\tau,\eta, \sigma$. The likelihood is $p(y,X|\tau,\eta,\sigma) = N(y|0,K_{XX}(\tau,\eta) + I\sigma^2)$, where $[K_{XX}]_{i,j} = \tau \exp(-\frac{1}{2\eta^2}\|x_i-x_j\|^2)$, and we use standard Gaussian priors $\sigma,\eta,\tau \sim N(0,1)$.

```python
# Helper to create log likelihood for GPR
from FSM.utils.gpr import logpdf_gp_fn, generate_linear_XY

key = jax.random.PRNGKey(42)
n_samples = 500
X, y, key = generate_linear_XY(key, n_samples, x_min=-3.0, x_max=3.0)

print("X shape:", X.shape)
print("Y shape:", y.shape)

# Define log-posterior of covariance hyperparameters
logprob_fn = jax.jit(get_logpdf_fn(y, X))

```

We instantiate the FSM and define the `step` function:

```python
# FSM construction
fsm = NutsFSM(
              step_size = 0.01,
              max_num_expansions=10,
              divergence_threshold=1000,
              inverse_mass_matrix=jnp.eye(3),
              logprob_fn
             )
step = jax.vmap(jax.jit(fsm.step))
```
We initialize the prng keys, algorithm state ($k=0$) and inputs ($z$) (the latter using the .init() method, which is called on $x = $`init_pos` and `init_rng`

```python
# RNG init
rng = jrnd.PRNGKey(42)
init_rng, pos_rng, chain_rng, rng = jrnd.split(rng, 4)

# Initializing position (x), alg state (k=0) and block inputs (z) (for 128 chains).
init_pos = jrnd.normal(pos_rng, (128, 3))
alg_state = jnp.zeros((128,), dtype=int)
init_inputs = jax.vmap(fsm.init)(init_rng, init_pos)
```
We run the FSM for 1000 samples.
```python
# Running and storing
start = time()
samples, _ = jax.block_until_ready(
    get_fsm_samples(alg_state, init_inputs, num_chains=128, num_samples=1000, step_fn = step)
)
print(time()-start)
```


## Reproducing Experiments
Once the repo is git cloned locally, you can re-run the experiments in the paper using the below commands. Note that results and speed-ups using our FSM may depend on the available memory and hardware. We ran our experiments on an NVIDIA A100 GPU with 32GB Memory.

```bash
# Run Delayed Rejection Experiment 7.1
python src/experiments/DR/GaussDR.py

# Run Elliptical Slice Experiment 7.2
python src/experiments/GPR/GPR_ESS_UCI_in.py # without amortization
python src/experiments/GPR/GPR_ESS_UCI_in.py # with amortization

# Run NUTS Experiment 7.3
python src/experiments/CorrG/corrg_nuts.py

# Run NUTS/TESS Experiment 7.4
python src/experiments/NUTS_benchmarks/nuts_benchmarks.py

# Run NUTS additional benchmarks
python src/experiments/NUTS_benchmarks/nuts_gpr.py
python src/experiments/NUTS_benchmarks/nuts_posteriordb.py
```

## Citation
If you use JAX-FSM-MCMC or the accompanying ICML 2025 paper in your work, please cite it as:

```bibtex
@inproceedings{dance2025efficiently,
  title     = {Efficiently Vectorized MCMC on Modern Accelerators},
  author    = {Dance, H., Glaser, P., Orbanz, P. and Adams, R.P.},
  booktitle = {Proceedings of the 42nd International Conference on Machine Learning (iCML 2025)},
  year      = {2025},
  url       = {https://github.com/hwdance/jax-fsm-mcmc},
}
```

## License
This project is licensed under the Apache 2.0 License. See the LICENSE file for details.

## Contact
For questions or feedback, please contact:

Hugh W. Dance,
PhD Researcher, Machine Learning,
Gatsby Computational Neuroscience Unit, UCL
uctphwd@ucl.ac.uk


