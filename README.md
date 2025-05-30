# JAX-FSM-MCMC: Vectorized MCMC without synchronization barriers

This repository contains JAX implementations of several stochastic-length proposal MCMC algorithms (i.e. HMC-NUTS and Slice sampling variants) for more efficient execution on SIMD architectures, when vectorizating with `vmap`. The implementation method is based on the paper "Efficiently Vectorized MCMC on Modern Accelerators" [https://www.arxiv.org/abs/2503.17405](url). 

### TL;DR

- **Issue:** Wrapping a data-dependent `while` loop in `jax.vmap` produces a single XLA `While` over the whole batch, so each iteration stalls until **all** batch elements finish—creating a full-batch synchronization barrier.

- **Solution:** We split the computation at each loop boundary into separate blocks, then use `jax.lax.switch` or `jax.lax.cond` to dispatch each block until every chain recovers its required samples. This lets each Markov chain progress through its own state sequence independently, eliminating the synchronization barrier.

### Why the `vmap` + `while` Barrier Occurs

When you wrap a Python `while` loop (or any `lax.while_loop`) in `jax.vmap`, JAX lowers your loop to a single XLA `While` operation whose body processes the *entire* batch in lockstep. Each iteration of the loop cannot proceed until *all* batch elements have finished their current iteration, creating an implicit synchronization barrier:

```python
import jax
import jax.numpy as jnp

def f(x):
    # A simple data-dependent loop
    def cond(state):
        i, x = state
        return i < 10
    def body(state):
        i, x = state
        return i + 1, x + jnp.sin(x)
    return jax.lax.while_loop(cond, body, (0, x))[1]

batched_f = jax.vmap(f)
```

Under the hood, this compiles to one XLA `While` op over a shape `[batch, ...]` tensor, enforcing that all batch lanes advance in lockstep. This behavior is not a JAX bug, but rather a consequence of how Python control flow is traced and lowered into XLA HLO:

* **Single HLO loop**: The entire loop (for all inputs) becomes one XLA op.
* **Batched execution**: `vmap` lifts that op to operate on a batched shape, so barrier semantics follow naturally.

Unless XLA introduces per-lane streaming loops, any data-dependent loop under `vmap` will serialize across the batch dimension.

## Our Solution: FSM-Based Desynchronization

To avoid the per-iteration sync barrier when vectorizing data-dependent loops, we model the single-chain `sample` kernel as a Finite State Machine (FSM) and drive it using a lightweight `step` function under `vmap`.

### 1. FSM Construction
- **Block decomposition**: Split the original `sample` code at each `while`-loop boundary into K contiguous blocks `S_1, ..., S_K`.  
- **States & transitions**: Each block `S_k` is a pure function on program state `z`, and a transition function `delta(k, z)` determines the next block based on loop-termination conditions.

### 2. Runtime `step` Function
```python
def step(k, z):
    is_sample = (k == K)            # flag when final block executes
    z = lax.switch(k, [S_1, ..., S_K], z)
    k = lax.switch(k, [delta(1, z), ..., delta(K, z)])
    return k, z, is_sample

## Performance Considerations

This section explains why combining `jax.vmap` with data-dependent loops introduces a synchronization barrier and offers guidance on writing efficient, batched control flows.

### 1. Why the `vmap` + `while` Barrier Occurs

When you wrap a Python `while` loop (or any `lax.while_loop`) in `jax.vmap`, JAX lowers your loop to a single XLA `While` operation whose body processes the *entire* batch in lockstep. Each iteration of the loop cannot proceed until *all* batch elements have finished their current iteration, creating an implicit synchronization barrier:

```python
import jax
import jax.numpy as jnp

def f(x):
    def cond(state):
        i, x = state
        return i < 10
    def body(state):
        i, x = state
        return i + 1, x + jnp.sin(x)
    return jax.lax.while_loop(cond, body, (0, x))[1]

batched_f = jax.vmap(f)
```

Under the hood, this compiles to one XLA `While` op over a shape `[batch, ...]` tensor, enforcing that all batch lanes advance in lockstep.

### 2. Fundamental Nature

This behavior is not a JAX bug—it's a consequence of how Python control flow is traced and lowered into XLA HLO:

* **Single HLO loop**: The entire loop (for all inputs) becomes one XLA op.
* **Batched execution**: `vmap` lifts that op to operate on a batched shape, so barrier semantics follow naturally.

Unless XLA introduces per-lane streaming loops, any data-dependent loop under `vmap` will serialize across the batch dimension.

### 3. Runtime Driver

To generate **N** samples per chain without intermediate barriers, we implement a simple driver that repeatedly calls `vmap(step)` until each chain has produced N samples:

```python
import jax
import jax.numpy as jnp
from jax import vmap

# step: (k, z) -> (k, z, is_sample)

def runtime_driver(z0, k0, N):
    """
    z0: initial states for B chains
    k0: initial block indices for B chains
    N: desired number of samples
    """
    z = z0
    k = k0
    samples = []
    
    while len(samples) < N:
        # advance each chain by one block
        k, z, is_sample = vmap(step)(k, z)
        # collect states for chains that finished a sample
        samples.append(jnp.where(is_sample[:, None], z, jnp.zeros_like(z)))
    
    # stack over the sampling dimension: shape [N, B, ...]
    return jnp.stack(samples, axis=0)
```

This loop only synchronizes when stacking the final array of samples, eliminating per-iteration barriers.





