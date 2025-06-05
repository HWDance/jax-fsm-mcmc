"""
Wrapper to execute BlackJax MCMC samplers
"""
import jax
import jax.numpy as jnp
import blackjax
from blackjax.diagnostics import effective_sample_size
from time import time

def run_blackjax(
    kernel,
    rng, position, input_dims,
    num_samples, num_chains, burn_in = 0,
    batch_fn = jax.vmap,
    initial_compile = True,
):
    
    def inference_loop_multiple_chains(rng_key, kernel, initial_state, num_samples, num_chains):
    
        @jax.jit
        def one_step(states, rng_key):
            keys = jax.random.split(rng_key, num_chains)
            states, info = batch_fn(kernel)(keys, states)
            return states, (states, info)
    
        keys = jax.random.split(rng_key, num_samples)
        _, (states, info) = jax.lax.scan(one_step, initial_state, keys)
    
        return states, info

    # Initialising
    initial_state = batch_fn(kernel.init)(position)

    if initial_compile:
        # Removing compile time for inner step fn
        _ = inference_loop_multiple_chains(rng, kernel.step, initial_state, 1, num_chains)


    # Running
    start = time()
    samples, info = jax.block_until_ready(
                inference_loop_multiple_chains(
                    rng, 
                    kernel.step, 
                    initial_state, 
                    num_samples, 
                    num_chains
                )
        )
    walltime = time() - start

     # Diagnostics
    ess = effective_sample_size(
        samples.position[burn_in:],
        chain_axis = 1,
        sample_axis = 0
    )

    # Storing
    return walltime, ess, info