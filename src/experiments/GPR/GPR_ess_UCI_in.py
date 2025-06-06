# Imports
import sys
import jax
import jax.numpy as jnp
import jax.random as jrnd
import pickle
from pathlib import Path
import blackjax

# Data imports
from ucimlrepo import fetch_ucirepo 
import pandas as pd

from FSM.mcmc.ellipticalslice_in_bundle import EllipticalSliceFSM
from FSM.base.run_fsm_in import run_fsm
from FSM.base.run_blackjax import run_blackjax
from FSM.utils.gpr import logpdf_gp_fn as get_logpdf_fn

def main(seed=0, data_id=477, num_chains=128, num_samples=10000, burn_in = 0, batch_fn = jax.vmap, num_train = None):

    """ Constructing sampler inputs (rng, data, logprob, pos) """    

    # RNG init
    rng = jrnd.PRNGKey(seed)
    pos_rng, rng = jrnd.split(rng, 2)
      
    # fetch dataset 
    data = fetch_ucirepo(id=data_id) 
      
    # Determine ndata to use
    if num_train is None:
        num_train = len(jnp.array(data.data.targets)[:,0])
      
    # data extraction
    X = jnp.array(data.data.features)[:num_train]
    y = jnp.array(data.data.targets)[:num_train,0]

    # Standardising
    y = (y-y.mean())/y.std()
    X = (X-X.mean(0))/X.std(0)

    # Logprobfn
    logprob_fn = jax.jit(get_logpdf_fn(y, X))
    
    # Initialisation
    input_dims = 3
    init_pos = jrnd.normal(pos_rng, shape = (num_chains, input_dims))
    
    """ ESS configs """
    mean = jnp.zeros(input_dims)
    cov = jnp.eye(input_dims)

    """ FSM sampler run """
    fsm = EllipticalSliceFSM(mean,cov,logprob_fn)
    fsm_time, fsm_ess, fsm_calls = run_fsm(
        fsm,
        rng,
        init_pos,
        input_dims,
        num_samples, 
        num_chains,
        burn_in,
        batch_fn,
    )
    
    """ Standard sampler run """
    kernel = blackjax.elliptical_slice(
        logprob_fn, 
        mean=mean, 
        cov=cov
    )
    naive_time, naive_ess, naive_info = run_blackjax(
        kernel,
        rng, 
        init_pos, 
        input_dims,
        num_samples, 
        num_chains,
        burn_in,
        batch_fn
    )

    """Storing results"""
    obj = {"fsm_times" : fsm_time,
           "naive_times" : naive_time,
           "fsm_ess" : fsm_ess,
           "naive_ess" : naive_ess,
           "fsm_logpdfcalls" : fsm_calls,
           "naive_logpdfcalls" : naive_info.subiter.max(1).sum() * num_chains,
           "iter_counts" : naive_info.subiter
          }
    
    return obj


if __name__ == "__main__":
    num_chains = [1,2,4,8,16,32,64,128,256,512,1024]
    results = []
    for n in num_chains:
        for s in range(5):
            results.append(main(seed=s))
    print(results)