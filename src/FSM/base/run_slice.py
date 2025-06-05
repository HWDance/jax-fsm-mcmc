"""
"Blackjax equivalent" implementation of standard slice sampler .
"""
import jax
import jax.numpy as jnp
import jax.random as jrnd
from functools import partial
from blackjax.diagnostics import effective_sample_size
from time import time

@partial(jax.jit, static_argnums=2)
def get_sample(rng, position, logprob_fn, step_size=1.0):

    # RNG splits
    thresh_rng, dir_rng, bracket_rng, rng = jrnd.split(rng,4)
        
    # Draw a random threshold
    u = jrnd.uniform(thresh_rng)
    threshold = jnp.log(u) + logprob_fn(position)

    # Draw a random direction
    direction = jrnd.normal(dir_rng, (position.shape[0],))
    direction = direction / jnp.linalg.norm(direction)
    
    # Initialize the bracket
    upper = jrnd.uniform(bracket_rng, minval=0.0, maxval=step_size)
    lower = upper - step_size

    # Initialize iteration counters
    upper_iter_count = 1
    lower_iter_count = 1
    shrink_iter_count = 0
    max_count = 100 # new
    
    # Step out the bracket upper
    def cond_upper(val):
        upper, iter_count = val
        pos = position + upper * direction
        logprob = logprob_fn(pos)
        return (logprob >= threshold)


    def body_upper(val):
        upper, iter_count = val
        upper = upper + step_size
        iter_count = iter_count + 1
        return upper, iter_count

    upper, upper_iter_count = jax.lax.while_loop(cond_upper, body_upper, (upper, upper_iter_count))

    # Step out the bracket lower
    def cond_lower(val):
        lower, iter_count = val
        pos = position + lower * direction
        logprob = logprob_fn(pos)
        return (logprob >= threshold)

    def body_lower(val):
        lower, iter_count = val
        lower = lower - step_size
        iter_count = iter_count + 1
        return lower, iter_count

    lower, lower_iter_count = jax.lax.while_loop(cond_lower, body_lower, (lower, lower_iter_count))
    
    # Shrink the bracket and sample
    accept = False
    alpha = 0.0  # Initial value; will be updated in the loop
    
    def cond_shrink(val):
        rng, lower, upper, alpha, accept, iter_count = val
        return ~accept

    def body_shrink(val):
        rng, lower, upper, alpha, accept, iter_count = val
        alpha_rng, rng = jrnd.split(rng)
        alpha = jrnd.uniform(alpha_rng, minval=lower, maxval=upper)
        pos = position + alpha * direction
        logprob = logprob_fn(pos)
        accept = (logprob >= threshold)

        # Shrink the bracket if not accepted
        upper = jnp.where((alpha > 0) & (~accept), alpha, upper)
        lower = jnp.where((alpha < 0) & (~accept), alpha, lower)

        iter_count = iter_count + 1
        return rng, lower, upper, alpha, accept, iter_count

    val = (rng, lower, upper, alpha, accept, shrink_iter_count)
    rng, lower, upper, alpha, accept, shrink_iter_count = jax.lax.while_loop(cond_shrink, body_shrink, val)
    
    # Compute the new position
    new_pos = position + alpha * direction

    # Total iteration count
    total_iter_count = upper_iter_count + lower_iter_count + shrink_iter_count

    return rng, new_pos, total_iter_count

def sample_chain(rng, init_pos, num_samples, logprob_fn, step_size = 1.0):
    def scan_fn(carry, rng_input):
        pos, rng = carry 
        rng_step = rng 
        rng_step, new_pos, iter_count = get_sample(rng_step, pos, logprob_fn, step_size)
        return (new_pos, rng_step), (new_pos, iter_count) 
    
    rngs = jrnd.split(rng, num_samples)
    _, (samples, iter_counts) = jax.lax.scan(scan_fn, (init_pos,rng), rngs)
    return samples, iter_counts

def run_chains(rng, init_pos, num_samples, num_chains, logprob_fn, step_size = 1.0, batch_fn = jax.vmap):
    rngs = jrnd.split(rng, num_chains)
    samples_and_counts = batch_fn(sample_chain, in_axes=(0, 0, None, None, None))(
        rngs, 
        init_pos, 
        num_samples, 
        logprob_fn, 
        step_size
    )
    samples, iter_counts = samples_and_counts
    return samples, iter_counts


def run(
    logprob_fn, step_size,
    rng, position, input_dims,
    num_samples, num_chains, burn_in = 0,
    batch_fn = jax.vmap,
    initial_compile = True,
):
    
    # Initialisation
    init_rng, rng = jrnd.split(rng)
    init_pos = jrnd.normal(init_rng, (num_chains, input_dims))

    if initial_compile:        
        # Removing compile time for inner step function
        _ = jax.vmap(get_sample, in_axes = (0, 0, None, None))(
            jrnd.split(rng,num_chains), 
            init_pos, 
            logprob_fn, 
            step_size,
        )
        
    # Running
    start = time()
    samples, iter_counts = jax.block_until_ready(
        run_chains(
            rng, 
            init_pos, 
            num_samples, 
            num_chains, 
            logprob_fn,
            step_size,
            batch_fn
        )
    )
    walltime = time() - start
    logpdf_calls = iter_counts.max(0).sum() * num_chains

    # Diagnostics
    ess = effective_sample_size(
        samples[:,burn_in:],
        chain_axis = 0,
        sample_axis = 1
    )

    # logpdf_calls
    return walltime, ess, logpdf_calls, iter_counts