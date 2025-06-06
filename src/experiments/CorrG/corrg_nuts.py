# General imports
import sys
import os
import jax
import jax.numpy as jnp
import jax.random as jrnd
from jax.tree_util import tree_map
import numpy as np
import blackjax
from blackjax.diagnostics import effective_sample_size
from time import time

# FSM imports
from FSM.mcmc.nuts_bundle import NutsFSM
from FSM.base.run_fsm_in import jitted_update
from FSM.base.run_fsm_in import get_fsm_samples_chain as get_fsm_samples
from FSM.utils.correlated_MVN import get_logpdf_fn

""" Configs """
chain_list = [1,5,20,100,500]
input_dims = 100
step_size = 0.01 * 10 ** 0.5 
mala_step = 1e-3 
init_scale = 10 
max_num_expansions = 10
divergence_threshold = 1000
inverse_mass_matrix = jnp.eye(input_dims)
""""""

def main(seed = 0, runs = 5, corr=0.99, num_samples = 1000):
    
    fsm_time = []
    fsm_ess = []
    fsm_nsamples = []
    naive_time = []
    naive_ess = []
    naive_nsamples = []
    mala_time = []
    mala_ess = []
    mala_nsamples = []
    iter_counts = []

    logprob_fn = jax.jit(get_logpdf_fn(corr, input_dims))
    
    for r in range(runs):   
        for num_chains in chain_list:
            """ Blackjax nuts https://blackjax-devs.github.io/blackjax/examples/howto_sample_multiple_chains.html """
        
            rng = jrnd.PRNGKey(seed+r)
            init_rng, chain_rng, rng = jrnd.split(rng, 3)
            init_pos = jrnd.normal(init_rng, (num_chains, input_dims)) * init_scale
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
            naive_time.append(time()-start)
            naive_nsamples.append(num_samples*num_chains)
            iter_counts.append(iters)
                        
            """ Blackjax mala https://blackjax-devs.github.io/blackjax/examples/howto_sample_multiple_chains.html """
    
            rng = jrnd.PRNGKey(seed+r)
            init_rng, chain_rng, rng = jrnd.split(rng, 3)
            init_pos = jrnd.normal(init_rng, (num_chains, input_dims)) * init_scale
            
            # Build the kernel
            mala = blackjax.mala(logprob_fn, mala_step)
            
            # Initialize the state for multiple chains
            initial_state = jax.vmap(mala.init)(init_pos)
            
            def inference_loop_multiple_chains(rng_key, kernel, initial_state, num_samples, num_chains):
            
                @jax.jit
                def one_step(states, rng_key):
                    keys = jax.random.split(rng_key, num_chains)
                    states, info = jax.vmap(kernel)(keys, states)
                    return states, (states, info.acceptance_rate)
            
                keys = jax.random.split(rng_key, num_samples)
                _, (states, accept) = jax.lax.scan(one_step, initial_state, keys)
            
                return states, accept
        
            start = time()
            samples_bjmala, accept_rate_mala = jax.block_until_ready(
                    inference_loop_multiple_chains(rng, mala.step, initial_state, num_samples, num_chains)
            )
            mala_time.append(time()-start)
            mala_nsamples.append(num_samples*num_chains)
                    
        
            """ FSM """
        
            # RNG init
            rng = jrnd.PRNGKey(seed+r)
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
            init_pos = jrnd.normal(init_rng, (num_chains, input_dims)) * init_scale
            init_rng = jrnd.split(chain_rng, num_chains)
            fsm_idx = jnp.zeros((num_chains,), dtype=int)
            states = jax.vmap(fsm.init)(init_rng, init_pos)
            
            # Running and storing
            start = time()
            samples, _ = jax.block_until_ready(
                get_fsm_samples(fsm_idx, states, num_chains, num_samples, input_dims, step_fn)
            )
            fsm_time.append(time()-start)
            fsm_nsamples.append(len(samples)*num_chains)
            
        for num_chains in chain_list:
            ess_bjmala = effective_sample_size(samples_bjmala.position, chain_axis = 1,sample_axis = 0)
            ess_bj = effective_sample_size(samples_bj.position, chain_axis = 1,sample_axis = 0)
            ess = effective_sample_size(samples, chain_axis = 1,sample_axis = 0)
            naive_ess.append(ess_bj/chain_list[-1] * num_chains)
            mala_ess.append(ess_bjmala/chain_list[-1] * num_chains)
            fsm_ess.append(ess/chain_list[-1] * num_chains)

    fsm_ess_per_second = np.zeros((runs, len(chain_list)))
    naive_ess_per_second = np.zeros((runs, len(chain_list)))
    mala_ess_per_second = np.zeros((runs, len(chain_list)))
    trial = 0
    for r in range(runs):
        for c in range(len(chain_list)):
            fsm_ess_per_second[r,c] = fsm_ess[trial].mean() / fsm_time[trial]
            naive_ess_per_second[r,c] = naive_ess[trial].mean() / naive_time[trial]
            mala_ess_per_second[r,c] = mala_ess[trial].mean() / mala_time[trial]
            trial += 1

    return {
        "fsm_ess" : fsm_ess,
        "naive_ess" : naive_ess,
        "mala_ess" : mala_ess,
        "fsm_ess_per_second" : fsm_ess_per_second,
        "naive_ess_per_second" : naive_ess_per_second,
        "mala_ess_per_second" : mala_ess_per_second,
        "samples_bj" : samples_bj,
        "samples_bjmala" : samples_bjmala,
        "iter_counts" : iter_counts,
        "accept_rate" : accept_rate,
        "accept_rate_mala" : accept_rate_mala,      
    }

if __name__ == "__main__":
    main()