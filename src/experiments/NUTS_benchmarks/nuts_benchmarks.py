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

# FSM imports
from FSM.mcmc.nuts_bundle import NutsFSM
from FSM.base.run_fsm_in import jitted_update
from FSM.base.run_fsm_in import get_fsm_samples_chain as get_fsm_samples

# Dataset imports
from distributions import BioOxygen, RegimeSwitchHMM, HorseshoeLogisticReg, PredatorPrey
import pandas as pd 

fsm_time = []
fsm_ess = []
fsm_nsamples = []
bj_time = []
bj_ess = []
bj_nsamples = []
iter_counts = []

def run(seed = 0, num_chains = 128, max_num_expansions = 10, divergence_threshold = 1000, num_samples = 1000, dataset = "google"):

    """ dataset import and cleaning """
    
    if dataset == "BOD":
        input_dims = 2
        print("Generating synthetic data...")
        N = 20
        theta0 = 1.
        theta1 = .1
        var = 2 * 10 ** (-4)
        times = jnp.arange(1, 5, 4/N)
        std_norms = jrnd.normal(jrnd.PRNGKey(seed), (N,))
        obs = theta0 * (1. - jnp.exp(-theta1 * times)) + jnp.sqrt(var) * std_norms
        
        print("Setting up Biochemical oxygen demand density...")
        dist = BioOxygen(times, obs, var)
    
    
    if dataset == "google":
        input_dims = 9
        print("Loading Google stock data...")
        data = pd.read_csv('google.csv')
        y = data.dl_ac.values * 100
        T, _ = data.shape
        
        print("Setting up Regime switching hidden Markov model...")
        dist = RegimeSwitchHMM(T, y)
    
    if dataset == "germancredit":
        input_dims = 51
        print("Loading German credit data...")
        data = pd.read_table('german.data-numeric', header=None, delim_whitespace=True)
        ### Pre processing data as in NeuTra paper
        y = -1 * (data.iloc[:, -1].values - 2)
        X = data.iloc[:, :-1].apply(lambda x: -1 + (x - x.min()) * 2 / (x.max() - x.min()), axis=0).values
        X = np.concatenate([np.ones((1000, 1)), X], axis=1)
        N_OBS, N_REG = X.shape
    
        N_PARAM = N_REG * 2 + 1
        print("\n\nSetting up German credit logistic horseshoe model...")
        dist = HorseshoeLogisticReg(X, y)
    
    if dataset == "predatorprey":
        input_dims = 8
        print("Loading predator-prey data...")
        data = pd.read_table("lynxhare.txt", sep=" ", names=['year', 'prey', 'pred', ''])
    
        print("Setting up predator-prey model...")
        dist = PredatorPrey(
            data.year.values, data.pred.values, data.prey.values,
        )
    
    """ logprob constructon """
    
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
    
    #flat_x, unravel_x = ravel_pytree(dist.init_params)
    
    # Def logprob that acts on flattened x
    def logprob_fn(x):
        x = unravel_x(x)
        return dist.logprob_fn(x)
    
    # Test logprob on flat_pos for num_chains
    dist.initialize_model(rng, num_chains)
    flat_pos = jax.vmap(flatten_chain)(dist.init_params)
    jax.vmap(logprob_fn)(flat_pos)
    
    """ Warm-up"""
    
    start = time()
    rng_key = jrnd.PRNGKey(seed)
    warmup = blackjax.window_adaptation(blackjax.nuts, logprob_fn)
    rng_key, warmup_key, sample_key = jax.random.split(rng_key, 3)
    (state, parameters), _ = warmup.run(warmup_key, flat_x, num_steps=400)
    print(parameters)
    print(time() - start)
    step_size = parameters['step_size']
    inverse_mass_matrix = jnp.diag(parameters['inverse_mass_matrix'])#jnp.eye(input_dims)
    
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
        get_fsm_samples(fsm_idx, states, num_chains, num_samples, input_dims, step_fn)
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
    