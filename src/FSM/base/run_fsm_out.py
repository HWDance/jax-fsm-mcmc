""" 
Amortized FSM Runtime ( i.e. all computations "outside" step_fn).

get_fsm_updates: essentially takes the `step' of an FSM and jit-compiles it to update blocks 
    of size m<n (n = # samples). Default choice is m = min(100,nsamples).
    Note each call of step_fn should execute fsm transitions until a log-pdf call is needed
    for a batch-member, at which point logpdf is updated as req_pos. 

get_fsm_samples: executes get_fsm_updates until n_tot-samples recovered (no per-chain sample
    counts enforced). Will produce biased expectations unless reweighting samples by n-per chain. 

get_fsm_samples_chain: executes get_fsm_updates until n-samples per chain recovered.

get_fsm_samples_chain_dict: same as get_fsm_samples_chain but handles dicts instead of jax arrays

run_fsm: outer wrapper used in experiments. 
"""
# Imports
import jax
import jax.numpy as jnp
import jax.random as jrnd
import numpy as np
from blackjax.diagnostics import effective_sample_size
from time import time

# Update function
def get_fsm_updates(
    fsm_indices, 
    states, 
    logprobs, 
    num_steps, 
    logprob_fn, 
    step_fn
):
    
    def _scan_body(carry, _):
        fsm_indices, states, logprobs = carry
        (
         fsm_indices, 
         states, 
         req_pos, 
         mask, 
         samples
        ) = step_fn(fsm_indices, states, logprobs)
        logprobs = logprob_fn(req_pos)
        
        return (fsm_indices, states, logprobs), (samples, mask)

    return jax.lax.scan(_scan_body,(fsm_indices, states, logprobs), None, length=num_steps)

jitted_update = jax.jit(get_fsm_updates, static_argnames = ['num_steps', 'logprob_fn', 'step_fn'])

    
# function to recover at least K samples of fsm
def get_fsm_samples(
    init_indices, 
    init_states, 
    init_logprobs, 
    num_chains, 
    num_samples, 
    dims, 
    logprob_fn = None, 
    step_fn = None,
    num_steps = 100,
):
    
    # Move to CPU+numpy for fast slicing
    def quick_slice(array,mask):
        return np.array(array)[np.array(mask)]
    
    # Initialisation for outer while loop
    fsm_indices  = init_indices
    states = init_states
    logprobs = init_logprobs
    samples = [] 
    sample_count = 0
    step_count = 0
    logpdf_calls = 0

    # Outer while loop using a large number of samples
    while sample_count < num_samples*num_chains:

        # Run the scan function for estimated number of steps
        (fsm_indices, states, logprobs), (new_samples, new_mask)  = jitted_update(
            fsm_indices, 
            states,
            logprobs,
            num_steps,
            logprob_fn, 
            step_fn
        ) # run scan for num_steps
    
        # Store accepted samples
        accepted_samples = quick_slice(new_samples, new_mask)
        samples.append(accepted_samples)
        
        # Update metrics
        sample_count += len(accepted_samples)
        step_count += num_steps
        
    logpdf_calls = num_chains*step_count

    # Return concatenated samples
    return jnp.array(np.concatenate(samples)), logpdf_calls

def get_fsm_samples_chain(
    init_indices, 
    init_states,
    init_logprobs, 
    num_chains, 
    num_samples, 
    dims, 
    logprob_fn = None, 
    step_fn = None,
    num_steps = 100,
):
    
    # Initialisation for outer while loop
    fsm_indices  = init_indices
    states = init_states
    logprobs = init_logprobs
    samples = [] 
    sample_mask = []
    chain_counts = np.zeros(num_chains)
    step_count = 0


    # Outer while loop using a large number of samples
    while chain_counts.min() < num_samples:

        # Run the scan function for estimated number of steps
        (fsm_indices, states, logprobs), (new_samples, new_mask)  = jitted_update(
            fsm_indices, 
            states,
            logprobs,
            num_steps,
            logprob_fn,
            step_fn
        ) # run scan for num_steps
    
        # Store samples
        samples.append(new_samples)
        sample_mask.append(new_mask)
        
        # Update metrics
        chain_counts += new_mask.sum(0)
        step_count += num_steps
        
    logpdf_calls = step_count * num_chains
    
    # Create aggregated arrays + convert to numpy (fast slicing)
    total_samples, total_mask = np.concatenate(samples), np.concatenate(sample_mask)
    accepted_indices = np.argsort(~total_mask, axis=0, stable = True)[:num_samples]
    accepted_samples = np.take_along_axis(total_samples, accepted_indices[:, :, None], axis=0)

    # Return concatenated samples
    return accepted_samples, logpdf_calls

def get_fsm_samples_chain_dict(
    init_indices, 
    init_states,
    init_logprobs, 
    num_chains, 
    num_samples, 
    logprob_fn,
    step_fn,
    num_steps=100,
    dict_keys = None,
):

    # Initialisation for outer while_loops
    fsm_indices = init_indices
    states = init_states
    logprobs = init_logprobs
    samples_store = {k: [] for k in dict_keys} 
    mask_store = []
    chain_counts = np.zeros(num_chains, dtype=int)
    step_count = 0

    # Outer while loop: keep scanning until each chain has num_samples
    while chain_counts.min() < num_samples:
        
        # Run the scan function for number of steps
        (fsm_indices, states, logprobs), (new_samples, new_mask)  = jitted_update(
            fsm_indices, 
            states,
            logprobs,
            num_steps,
            logprob_fn,
            step_fn
        )
        # new_samples is a dict of arrays [num_steps, num_chains, ...]
        # new_mask is a bool array [num_steps, num_chains]

        # Convert from JAX to NumPy (if needed) and store
        for k, arr in new_samples.items():
            samples_store[k].append(np.array(arr))  # shape = [num_steps, num_chains, ...]
        mask_store.append(np.array(new_mask))       # shape = [num_steps, num_chains]

        # Update chain_counts
        chain_counts += new_mask.sum(axis=0)  # how many accepted in each chain
        step_count += num_steps

    # We used step_count * num_chains logprob calls (approx)
    logpdf_calls = step_count * num_chains

    # Concatenate all partial arrays along axis=0 (the "time" dimension)
    # shape => [T_total, num_chains, ...]
    total_samples = {}
    for k, samples_store_k in samples_store.items():
        total_samples[k] = np.concatenate(samples_store_k, axis=0)
    total_mask = np.concatenate(mask_store, axis=0)  # shape => [T_total, num_chains]

    # Next, figure out the first `num_samples` accepted indices per chain
    # We can do that by sorting on ~mask so that "True" appears first
    # Then taking the top num_samples for each chain
    accepted_indices = np.argsort(~total_mask, axis=0, kind="stable")[:num_samples, :]

    # Now we gather from total_samples[k] along axis=0
    # total_samples[k] shape => [T_total, num_chains, (...)]
    accepted_samples = {}
    for k, arr in total_samples.items():
        # Insert a new axis for the final dims so we can gather
        # shape of accepted_indices => [num_samples, num_chains]
        # shape of arr => [T_total, num_chains, (...)]
        # np.take_along_axis => match on axis=0
        # We add None to expand dimensions so it matches the trailing dims
        if arr.ndim == 2:
            # If this is supposed to be [T, 1], reshape it
            arr = arr.reshape(arr.shape[0],arr.shape[1], 1)        
        gathered = np.take_along_axis(
            arr,
            accepted_indices[..., None],  # shape => [num_samples, num_chains, 1, ...]
            axis=0,
        )
        # => shape => [num_samples, num_chains, (...)]
        accepted_samples[k] = gathered

    return accepted_samples, logpdf_calls
    
def run_fsm(
    fsm, logprob_fn, 
    rng, position, input_dims,
    num_samples, num_chains, burn_in = 0,
    batch_fn = jax.vmap,
    initial_compile = True,
):

    # Getting batched functions
    step_fn = batch_fn(jax.jit(fsm.step))
    logprob_fn = batch_fn(logprob_fn)
    
    # Initialisation
    init_rng = jrnd.split(rng, num_chains)
    fsm_idx = jnp.zeros((num_chains,), dtype=int)
    logprobs = logprob_fn(position)
    states = jax.vmap(fsm.init)(init_rng, position)

    if initial_compile:        
        # Removing compile time for inner step function
        _ = jitted_update(fsm_idx, states, logprobs, 100, logprob_fn, step_fn)


    # Running
    start = time()
    samples, logpdf_calls = jax.block_until_ready(
            get_fsm_samples_chain(
                fsm_idx, 
                states, 
                logprobs,
                num_chains, 
                num_samples+burn_in, 
                input_dims, 
                logprob_fn,
                step_fn,
            )
        )
    walltime = time() - start

    # Diagnostics
    ess = effective_sample_size(
        samples[burn_in:],
        chain_axis = 1,
        sample_axis = 0
    )

    # Storing
    return walltime, ess, logpdf_calls