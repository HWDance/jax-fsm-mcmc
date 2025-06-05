""" 
Naive runtime of FSM (enforces synchronization per sample).

Should behave equivalently to standard MCMC implementations in Blackjax if
step_fn = all transition kernel work (as in e.g. Delayed-Rejection MH)
"""

# Imports
import jax
import jax.numpy as jnp
import jax.random as jrnd
import numpy as np
from blackjax.diagnostics import effective_sample_size
from time import time
from jax.tree_util import Partial

# Function to get a sample (new state)
def one_step(
    state, 
    step_fn,
):

    # Run FSM until we have a sample
    def cond_fn(loop_state):

        fsm_idx, state, have_sample, sample_loc = loop_state
        
        return ~have_sample        

    def body_fn(loop_state):

        fsm_idx, state, have_sample, sample_loc = loop_state
        new_state = step_fn(fsm_idx, state)

        return new_state

    # Init loopstate
    init_loop_state = (0, state, False, state.position)

    # Get sample
    loop_state = jax.lax.while_loop(cond_fn, body_fn, init_loop_state)

    # unpack final loop state
    fsm_idx, state, have_sample, sample_loc = loop_state    
    
    return state

jitted_one_step = jax.jit(one_step, static_argnames = ['step_fn'])

def inference_loop_multiple_chains(initial_state, num_samples, step_fn):
    
    def scan_body(carry,x):
        
        carry = jitted_one_step(carry, step_fn)
        
        return carry, (carry.position, carry.proposal_iter_sample)
    
    _, states = jax.lax.scan(scan_body, initial_state, xs = None, length = num_samples)

    return states