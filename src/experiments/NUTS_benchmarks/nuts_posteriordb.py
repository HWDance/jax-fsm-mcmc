# General imports
import os
import jax
import jax.numpy as jnp
import jax.random as jrnd
from jax.tree_util import tree_map
from jax.flatten_util import ravel_pytree
import numpy as np
import blackjax
from blackjax.diagnostics import effective_sample_size
from time import time
import pickle 
import pandas as pd 

# FSM imports
from FSM.mcmc.nuts_bundle import NutsFSM
from FSM.base.run_fsm_in import jitted_update
from FSM.base.run_fsm_in import get_fsm_samples_chain as get_fsm_samples

#from pilots import HierarchicalModel as Model
#from soil import TwoPoolModel as Model
from soil_diffrax import TwoPoolModel as Model
#from pilots import data
from soil_diffrax import data

fsm_time = []
fsm_ess = []
fsm_nsamples = []
bj_time = []
bj_ess = []
bj_nsamples = []
mala_time = []
mala_ess = []
mala_nsamples = []
iter_counts = []

def run(seed = 0, nu_chains = 128, max_num_expansions = 10, divergence_threshold = 1000, 
        dataset = "soil_diffrax", num_samples = 1000):

    """ logprob constructon """
    dist = Model(data)
    # Get params (vectorized for google stock)
    rng = jrnd.PRNGKey(seed)
    dist.initialize_model(rng, 1)
    
    # flatten non-vectorized variant to get unravel_fn
    flat_x, unravel_x = ravel_pytree(jax.tree_map(lambda x: x[0], dist.init_params))
    # function to flatten vectorized params
    def flatten_chain(pos):
                flat_pos, _ = ravel_pytree(pos)
                return flat_pos
    # check that flat_pos = vectorized flat_x
    flat_pos = jax.vmap(flatten_chain)(dist.init_params)
    assert((flat_pos == flat_x[None]).all())
    
    # Def logprob that acts on flattened x
    def logprob_fn(x):
        x = unravel_x(x)
        return dist.logprob_fn(x)
    
    start = time()
    rng_key = jrnd.PRNGKey(seed)
    warmup = blackjax.window_adaptation(blackjax.nuts, logprob_fn)
    rng_key, warmup_key, sample_key = jax.random.split(rng_key, 3)
    (state, parameters), _ = warmup.run(warmup_key, flat_x, num_steps=400)
    print(parameters)
    print(time() - start)
    step_size = parameters['step_size']
    inverse_mass_matrix = jnp.diag(parameters['inverse_mass_matrix'])
    
    """ FSM """
    # RNG init
    rng = jrnd.PRNGKey(seed)
    init_rng, chain_rng, rng = jrnd.split(rng, 3)
    
    # FSM construction
    fsm = NutsFSM(
                  step_size,
                  max_num_expansions,
                  divergence_threshold,
                  inverse_mass_matrix,
                  logprob_fn
                 )
    step_fn = jax.vmap(jax.jit(fsm.step))
    
    # Initialisation
    dist.initialize_model(init_rng, num_chains)
    init_pos =  dist.init_params
    flat_pos = jax.vmap(flatten_chain)(init_pos)
    init_rng = jrnd.split(chain_rng, num_chains)
    fsm_idx = jnp.zeros((num_chains,), dtype=int)
    states = jax.vmap(fsm.init)(init_rng, flat_pos)
    
    # Running and storing
    start = time()
    samples, _ = jax.block_until_ready(
        get_fsm_samples(fsm_idx, states, num_chains, num_samples, 1, step_fn)
    )
    fsm_time.append(time()-start)
    fsm_nsamples.append(len(samples)*num_chains)
    print(fsm_time)
    ess = effective_sample_size(samples, chain_axis = 1,sample_axis = 0)
    print(ess)
    results_fsm = {'time': fsm_time, 
                 'ess': ess}
    pickle.dump(results_fsm,
                open('fsm_time_step={0}_samples={1}_dataset={2}_tune={3}.pkl'.format(step_size, num_samples, dataset, tune), 'wb'))
    
    """ Blackjax nuts https://blackjax-devs.github.io/blackjax/examples/howto_sample_multiple_chains.html """
    rng = jrnd.PRNGKey(seed)
    init_rng, chain_rng, rng = jrnd.split(rng, 3)
    init_pos = flat_pos
    # Build the kernel
    
    nuts = blackjax.nuts(logprob_fn, step_size, inverse_mass_matrix)
    
    # Initialize the state for multiple chains
    initial_state = jax.vmap(nuts.init)(init_pos)
    
    def inference_loop_multiple_chains(rng_key, kernel, initial_state, num_samples, num_chains):
    
        @jax.jit
        def one_step(states, rng_key):
            keys = jax.random.split(rng_key, num_chains)
            states, info = jax.vmap(kernel)(keys, states)
            return states, (states, info.num_integration_steps, info.acceptance_rate, info.is_divergent, info.is_turning)
    
        keys = jax.random.split(rng_key, num_samples)
        _, (states, iters, accept, div, turn) = jax.lax.scan(one_step, initial_state, keys)
    
        return states, iters, accept, div, turn
    
    start = time()
    samples_bj, iters, accept_rate, is_divergent, is_turning = jax.block_until_ready(
            inference_loop_multiple_chains(rng, nuts.step, initial_state, num_samples, num_chains)
    )
    bj_time.append(time()-start)
    bj_nsamples.append(num_samples*num_chains)
    iter_counts.append(iters)
    print(bj_time, accept_rate.mean())
    ess_bj = effective_sample_size(samples_bj.position, chain_axis = 1,sample_axis = 0)
    print(ess_bj)
    results_bj = {"time" : bj_time, 
                 "accept_rate" : accept_rate.mean(),
                 'ess': ess_bj}
    pickle.dump(results_bj,
                open('bj_time_step={0}_samples={1}_dataset={2}.pkl'.format(step_size, num_samples, dataset), 'wb'))
    
     return results_fsm, results_bj   

if __name__ == "__main__":
    run() 