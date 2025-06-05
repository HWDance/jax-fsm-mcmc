# Imports
import jax
import jax.numpy as jnp
import jax.random as jrnd
import jax_dataclasses as jdc
from jax.scipy.linalg import cho_solve, cholesky
from functools import partial
import blackjax

from FSM.mcmc.delayedrejection_in import DelayedRejectionFSM
from FSM.mcmc.delayedrejection_in_uncondensed import DelayedRejectionFSM as DelayedRejectionFSMunrolled
from FSM.base.run_fsm_in import run_fsm, get_fsm_samples, get_fsm_samples_chain, get_fsm_updates
from FSM.base.run_fsm_in_naive import one_step, inference_loop_multiple_chains

from timeit import timeit, repeat
from time import time

def logprob_fn(x):
    return -(0.5*x**2).sum()

""" Configs """
scale = 0.01
maxiter = 100
fsm = DelayedRejectionFSMunrolled(scale, maxiter, logprob_fn)
fsm_compact = DelayedRejectionFSM(scale, maxiter, logprob_fn)
batch_fn = jax.vmap
""""""

def main(seed=0, runs=1, num_chains=128, num_samples = 10000, num_steps = 100, dims = 1):
    
    fsm_times = []
    fsm_compact_times = []
    naive_times = []

    for r in range(runs):
    
        # Init
        rng = jrnd.PRNGKey(seed+r)
        position = jrnd.normal(rng, (num_chains,dims))
    
        # Getting batched functions
        step_fn = batch_fn(jax.jit(fsm.step))
        step_fn_compact = batch_fn(jax.jit(fsm_compact.step))
        step_fn_naive = jax.jit(fsm_compact.step)
        
        # Initialisation
        init_rng = jrnd.split(rng, num_chains)
        fsm_idx = jnp.zeros((num_chains,), dtype=int)
        
        states = jax.vmap(fsm.init)(init_rng, position)
    
        # FSM run
        samples, _ = jax.block_until_ready(
            get_fsm_samples_chain(
                fsm_idx, 
                states, 
                num_chains, 
                num_samples, 
                dims = 1, 
                step_fn = step_fn,
                num_steps = num_steps,
            )
        )
        
        start = time()
        samples, _ = jax.block_until_ready(
            get_fsm_samples_chain(
                fsm_idx, 
                states, 
                num_chains, 
                num_samples, 
                dims = 1, 
                step_fn = step_fn,
                num_steps = num_steps,
            )
        )
        fsm_times.append(time()-start)
            
        # FSM compact run
        samples, _ = jax.block_until_ready(
            get_fsm_samples_chain(
                fsm_idx, 
                states, 
                num_chains, 
                num_samples, 
                dims = 1, 
                step_fn = step_fn_compact,
                num_steps = num_steps,
            )
        )
        
        start = time()
        samples, _ = jax.block_until_ready(
            get_fsm_samples_chain(
                fsm_idx, 
                states, 
                num_chains, 
                num_samples, 
                dims = 1, 
                step_fn = step_fn_compact,
                num_steps = num_steps,
            )
        )
        fsm_compact_times.append(time()-start)

        # Naive run
        vmapped_loop = jax.vmap(inference_loop_multiple_chains, in_axes = (0, None, None))
        jitted_loop = jax.jit(vmapped_loop, static_argnames = ['num_samples', 'step_fn'])
        
        samples, iters = jax.block_until_ready(
            vmapped_loop(
                states, 
                num_samples, 
                step_fn_naive,
            )
        )
        start = time()
        samples, iters = jax.block_until_ready(
            vmapped_loop(
                states, 
                num_samples, 
                step_fn_naive,
            )
        )
        naive_times.append(time()-start)

    return {"fsm_times" : fsm_times,
            "fsm_compact_times" : fsm_compact_times,
            "naive_times" : naive_times,
            "num_steps" : num_steps,
            "num_chains" : num_chains,
            "iters" : iters
           }


if __name__ == "__main__":
    main()
    print(iters.max(1).mean(), iters.max(0).mean())