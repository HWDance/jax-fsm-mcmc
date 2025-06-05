""" 
Non-Amortized FSM Runtime (i.e. all computations "inside" step_fn).

get_fsm_updates: essentially takes the `step' of an FSM and jit-compiles it to update blocks 
    of size m<n (n = # samples). Default choice is m = min(100,nsamples)

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
    num_steps, 
    step_fn
):
    
    def _scan_body(carry, _):
        fsm_indices, states = carry
        (
         fsm_indices, 
         states, 
         mask, 
         samples
        ) = step_fn(fsm_indices, states)
        
        return (fsm_indices, states), (samples, mask)

    return jax.lax.scan(_scan_body,(fsm_indices, states), None, length=num_steps)

jitted_update = jax.jit(get_fsm_updates, static_argnames = ['num_steps', 'step_fn'])

    
# function to recover at least K samples of fsm
def get_fsm_samples(
    init_indices, 
    init_states, 
    num_chains, 
    num_samples, 
    dims, 
    step_fn = None,
    num_steps = 100,
):
    
    # Move to CPU+numpy for fast slicing
    def quick_slice(array,mask):
        return np.array(array)[np.array(mask)]
    
    # Initialisation for outer while loop
    fsm_indices  = init_indices
    states = init_states
    samples = [] 
    sample_count = 0

    # Outer while loop using a large number of samples
    while sample_count < num_samples*num_chains:

        # Run the scan function for estimated number of steps
        (fsm_indices, states), (new_samples, new_mask)  = jitted_update(
            fsm_indices, 
            states,
            num_steps,
            step_fn
        ) # run scan for num_steps
    
        # Store accepted samples
        accepted_samples = quick_slice(new_samples, new_mask)
        samples.append(accepted_samples)
        
        # Update metrics
        sample_count += len(accepted_samples)

    # Return concatenated samples
    return jnp.array(np.concatenate(samples)), None

# function to recover at least K samples of fsm
def get_fsm_samples_chain(
    init_indices, 
    init_states, 
    num_chains, 
    num_samples, 
    dims = 1, 
    step_fn = None,
    num_steps = 100,
):
    
    # Initialisation for outer while loop
    fsm_indices  = init_indices
    states = init_states
    samples = [] 
    sample_mask = []
    chain_counts = np.zeros(num_chains)

    # Outer while loop using a large number of samples
    while chain_counts.min() < num_samples:

        # Run the scan function for estimated number of steps
        (fsm_indices, states), (new_samples, new_mask)  = jitted_update(
            fsm_indices, 
            states,
            num_steps,
            step_fn
        ) # run scan for num_steps
    
        # Store samples
        samples.append(new_samples)
        sample_mask.append(new_mask)
        
        # Update metrics
        chain_counts += new_mask.sum(0)

    # Create aggregated arrays + convert to numpy (fast slicing) for getting accepted samples per chain
    total_samples, total_mask = np.concatenate(samples), np.concatenate(sample_mask)
    accepted_indices = np.argsort(~total_mask, axis = 0, stable = True)[:num_samples]
    accepted_samples = np.take_along_axis(total_samples, accepted_indices[...,None],axis = 0)

    # Return concatenated samples
    return accepted_samples, None

def get_fsm_samples_chain_dict(
    init_indices, 
    init_states, 
    num_chains, 
    num_samples, 
    dims=1, 
    step_fn = None,
    num_steps = 100,
    dict_keys = None,
):
    
    # Initialisation for outer while loop
    fsm_indices  = init_indices
    states = init_states
    samples =  {k: [] for k in dict_keys} 
    sample_mask = []
    chain_counts = np.zeros(num_chains)

    # Outer while loop using a large number of samples
    while chain_counts.min() < num_samples:

        # Run the scan function for estimated number of steps
        (fsm_indices, states), (new_samples, new_mask)  = jitted_update(
            fsm_indices, 
            states,
            num_steps,
            step_fn
        ) # run scan for num_steps
    
        # Store samples
        for k, arr in new_samples.items():
            samples[k].append(np.array(arr))
        sample_mask.append(np.array(new_mask))
        
        # Update metrics
        chain_counts += new_mask.sum(0)

    # Create aggregated arrays + convert to numpy (fast slicing) for getting accepted samples per chain
    # Unlike the non dict version, we need to iterate through dict keys and merge the lists into arrays
    total_samples = {}
    for k, samples_store_k in samples.items():
        total_samples[k] = np.concatenate(samples_store_k, axis=0)
    total_mask = np.concatenate(sample_mask, axis=0)  # shape => [T_total, num_chains]

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
    
    # Return concatenated samples
    return accepted_samples, None

def run_fsm(
    fsm, 
    rng, position, input_dims,
    num_samples, num_chains, burn_in = 0,
    batch_fn = jax.vmap,
    initial_compile = True,
):

    # Getting batched functions
    step_fn = batch_fn(jax.jit(fsm.step))
    
    # Initialisation
    init_rng = jrnd.split(rng, num_chains)
    fsm_idx = jnp.zeros((num_chains,), dtype=int)
    
    states = jax.vmap(fsm.init)(init_rng, position)

    if initial_compile:
        # Removing compile time for inner step function
        _ = jitted_update(fsm_idx, states, 100, step_fn)

    # Running
    start = time()
    samples, _ = jax.block_until_ready(
            get_fsm_samples_chain(
                fsm_idx, 
                states, 
                num_chains, 
                num_samples+burn_in, 
                input_dims, 
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
    return walltime, ess, _