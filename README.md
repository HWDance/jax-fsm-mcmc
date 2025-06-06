# JAX-FSM-MCMC: Vectorized MCMC without synchronization barriers

This repository contains JAX implementations of several stochastic-length proposal MCMC algorithms (i.e. HMC-NUTS and Slice sampling variants) for more efficient execution on SIMD architectures, when vectorizating with `vmap`. The implementation method is based on the paper "Efficiently Vectorized MCMC on Modern Accelerators" [https://www.arxiv.org/abs/2503.17405](https://www.arxiv.org/abs/2503.17405). 

### TL;DR

- **Issue:** Wrapping a data-dependent `while` loop in `jax.vmap` produces a single batched `While` with an aggregeated termination condition for the whole batch, so each iteration stalls until **all** batch elements finish—creating a full-batch synchronization barrier.

- **Solution:** We split the computation at each loop boundary into separate blocks, then use `jax.lax.switch` or `jax.lax.cond` to dispatch each block until every chain recovers its required samples. This lets each Markov chain progress through its own state sequence independently, eliminating the synchronization barrier.

## Table of Contents

1. [Installation](#installation)  
2. [Getting Started](#getting-started)  
   - [Basic Usage](#basic-usage)  
   - [API Overview](#api-overview)  
3. [Reproducing Experiments](#reproducing-experiments)  
4. [Development & Contributing](#development--contributing)  
5. [License](#license) 

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
Below is a minimal example showing how to run a simple NUTS chain using the FSM approach. For more advanced usage (slice samplers, customization, etc.), see the API Overview section.


### Basic Usage 




