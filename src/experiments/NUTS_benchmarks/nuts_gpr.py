# General imports
import sys
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
from ucimlrepo import fetch_ucirepo 

# FSM imports
from FSM.mcmc.nuts_bundle import NutsFSM
from FSM.base.run_fsm_in import jitted_update, run_fsm
from FSM.base.run_fsm_in import get_fsm_samples_chain as get_fsm_samples
from FSM.base.run_blackjax import run_blackjax
from FSM.utils.gpr import logpdf_gp_fn as get_logpdf_fn

from distributions import BioOxygen, RegimeSwitchHMM, HorseshoeLogisticReg, PredatorPrey

fsm_time = []
fsm_ess = []
fsm_nsamples = []
bj_time = []
bj_ess = []
bj_nsamples = []
mala_time = []
mala_ess = []
mala_nsamples = []
iter_ounts = []

def run(seed = 0, num_chains = 128, max_num_expansions = 10, divergence_threshold = 1000, num_samples = 1000):

    """ logprob constructon """
    # fetch dataset 
    data_id = 477
    input_dims = 3
    data = fetch_ucirepo(id=data_id) 
    
    # data extraction
    X = jnp.array(data.data.features)
    y = jnp.array(data.data.targets)
    
    # Standardising
    y = ((y-y.mean())/y.std())[:,0]
    X = (X-X.mean(0))/X.std(0)
    
    # Logprobfn
    logprob_fn = jax.jit(get_logpdf_fn(y, X))
    
    # Single chain Initialisation
    start = time()
    input_dims = 3
    rng_key = jrnd.PRNGKey(seed)
    x = jrnd.normal(rng_key, shape = (input_dims,))
    warmup = blackjax.window_adaptation(blackjax.nuts, logprob_fn)
    rng_key, warmup_key, sample_key = jax.random.split(rng_key, 3)
    (state, parameters), _ = warmup.run(warmup_key, x, num_steps=400)
    print(parameters)
    print(time() - start)
    
    step_size = parameters['step_size']
    inverse_mass_matrix = jnp.diag(parameters['inverse_mass_matrix'])#jnp.eye(input_dims)
    
    """ FSM sampler run """
      
    # Initialisation
    rng = jrnd.PRNGKey(seed)
    pos_rng, rng = jrnd.split(rng, 2)
    init_pos = jrnd.normal(pos_rng, shape = (num_chains, input_dims))
    
    fsm = NutsFSM(step_size = step_size, 
                 inverse_mass_matrix = inverse_mass_matrix,
                max_num_expansions = max_num_expansions,
                divergence_threshold = divergence_threshold, 
                 logdensity_fn = logprob_fn)
    
    fsm_time, fsm_ess, fsm_calls = run_fsm(
       fsm,
       rng,
       init_pos,
       input_dims,
       num_samples, 
       num_chains,
    )                                   
    
    """ Standard sampler run """
    kernel = blackjax.nuts(
        logprob_fn, 
        step_size=step_size, 
        inverse_mass_matrix=inverse_mass_matrix
    )
    bj_time, bj_ess, bj_info = run_blackjax(
        kernel,
        rng, 
        init_pos, 
        input_dims,
        num_samples, 
        num_chains,
    )
    
    results_fsm = {'time': fsm_time, 
                 'ess': fsm_ess}
    
    results_bj = {'time': bj_time, 
                 'ess': bj_ess}

    pickle.dump(results_fsm,
                open('fsm_time_seed={0}_samples={1}_dataset={2}.pkl'.format(seed,step_size, num_samples, "gpr"), 'wb'))
    
    pickle.dump(results_bj,
                open('bj_time_seed={0}_samples={1}_dataset={2}.pkl'.format(seed,step_size, num_samples, "gpr"), 'wb'))
    
    return results_fsm, results_bj

if __name__ == "__main__":
    results = []
    for i in range(3):
        results.append(run(seed = i))