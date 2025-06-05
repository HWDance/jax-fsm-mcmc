"""
Delayed Rejection FSM with unrolled states (i.e. artificially uses 4 states) """

import jax
import jax.numpy as jnp
import jax.random as jrnd
import jax_dataclasses as jdc

@jdc.pytree_dataclass
class DelayedRejectionState:
  rng:       jax.Array
  position:   jax.Array
  proposal:   jax.Array
  proposal_iter: jax.Array
  proposal_iter_sample: jax.Array
  proposal_iter_max: jax.Array
  poslogprob:   jax.Array
  proplogprob:   jax.Array
  maxlogprob:    jax.Array

def log_clamped_ratio(x, y, z, epsilon=1e-10):
    """
    Compute log(min((max(0, exp(x) - exp(y)) / (exp(z) - exp(y))), 1))
    , assuming z > y
    """
    # Compute numerator max(epsilon, exp(x) - exp(y)) 
    numerator_clamped = jnp.maximum(0.0,jnp.exp(x) - jnp.exp(y))
    
    # Compute denominator = exp(z) - exp(y)
    denominator = jnp.exp(z) - jnp.exp(y)

    # Step 4: Compute ratio where denominator >0, else set to 0.0
    ratio = numerator_clamped / denominator

    return jnp.log(ratio)

def get_log_accept_prob(proplogprob, poslogprob, maxlogprob):
                
    return proplogprob -  poslogprob   

def get_log_accept_prob_2(proplogprob, poslogprob, maxlogprob):
                
    return log_clamped_ratio(proplogprob, maxlogprob, poslogprob)

    
class DelayedRejectionFSM:

    def __init__(
        self,
        scale,
        maxiter,
        logdensity_fn,
    ):
        DelayedRejectionFSM.scale = scale
        DelayedRejectionFSM.maxiter = maxiter
        DelayedRejectionFSM.logdensity_fn = logdensity_fn
    ################################################################################
    class PROPOSE:
        index: jax.Array = jnp.array(0, dtype = int)

        @classmethod
        def step(cls, state):
        
            with jdc.copy_and_mutate(state) as new_state:
            
                # seed
                proposal_rng, new_state.rng = jrnd.split(state.rng, 2)
                
                # Sample proposal
                new_state.proposal =  DelayedRejectionFSM.scale * jrnd.normal(
                    proposal_rng, 
                    shape = state.position.shape
                ) + state.proposal

            return (
                DelayedRejectionFSM.ACCEPT.index,
                new_state, # updated state
                False, # Sample not taken
                new_state.position # sample loc
            )

    class ACCEPT:
        index: jax.Array = jnp.array(1, dtype = int)
        
        @classmethod
        def step(cls, state):
        
            with jdc.copy_and_mutate(state) as new_state:

                # seed
                accept_rng, new_state.rng = jrnd.split(state.rng, 2)

                # Logprob of proposal
                new_state.proplogprob = DelayedRejectionFSM.logdensity_fn(new_state.proposal)
                
                # Acceptance probability
                log_accept_prob = jax.lax.cond(
                    state.proposal_iter <= 0,
                    get_log_accept_prob,
                    get_log_accept_prob_2,
                    new_state.proplogprob, new_state.poslogprob, new_state.maxlogprob,
                )
                
                # Uniform sample
                logu = jnp.log(jrnd.uniform(accept_rng))
                
                # Accept/Reject
                accept  = (logu < log_accept_prob) | (new_state.proposal_iter >= DelayedRejectionFSM.maxiter)
            return jax.lax.cond(
                accept,
                cls._done,
                cls._not_done,
                new_state,
            )
    
        @classmethod
        def _done(cls, state):
            ''' exit state '''
            return(
                DelayedRejectionFSM.DONE.index,
                state,
                False,
                state.position
            )


        @classmethod
        def _not_done(cls, state):
            ''' exit state '''
            return(
                DelayedRejectionFSM.UPDATE.index,
                state,
                False,
                state.position
            )
            
    class UPDATE:  
        index: jax.Array = jnp.array(2, dtype = int)
        
        @classmethod
        def step(cls, state):           
            
            with jdc.copy_and_mutate(state) as new_state: 

                # Overwrite logprobs for restart
                new_state.maxlogprob = jax.lax.cond(
                    state.proposal_iter == 0,
                    lambda x : state.proplogprob,
                    lambda x : jnp.maximum(state.maxlogprob,state.proplogprob),
                    None
                )

                # Update proposal iter
                new_state.proposal_iter = state.proposal_iter + 1
                
            return (
                DelayedRejectionFSM.PROPOSE.index,
                new_state, # updated state
                False, # Sample taken
                new_state.position # sample loc
            )
        
    class DONE:
        index: jax.Array = jnp.array(3, dtype = int)

        @classmethod
        def step(cls, state):
        
            with jdc.copy_and_mutate(state) as new_state:

                # If exceeded max proposals, restart position and poslogprob, else update
                new_state.position, new_state.poslogprob, sample_taken = jax.lax.cond(
                    new_state.proposal_iter >= DelayedRejectionFSM.maxiter,
                    lambda state : (state.position, state.poslogprob, True),
                    lambda state : (state.proposal, state.proplogprob, True),
                    new_state
                )

                # Update proposal iter
                new_state.proposal_iter_sample = state.proposal_iter
                new_state.proposal_iter = jnp.array(0,dtype = 'int32')
                    
                # Overwrite position for restart
                #new_state.position = new_state.proposal
                # Overwrite logprobs for restart
                #new_state.poslogprob = state.proplogprob
            
            return (
                DelayedRejectionFSM.PROPOSE.index,
                new_state, # updated state
                sample_taken, # Sample taken
                new_state.position # sample loc
            )
    
    ################################################################################
    
    # Define init function
    def init(self, rng, init_pos, iter_max = 10):
      return DelayedRejectionState(
        rng=rng,
        position=init_pos,
        proposal=init_pos,
        poslogprob=DelayedRejectionFSM.logdensity_fn(init_pos),
        proplogprob=DelayedRejectionFSM.logdensity_fn(init_pos),
        maxlogprob=jnp.array(-10**3, dtype = 'float32'),
        proposal_iter = jnp.array(0, dtype = 'int32'),
        proposal_iter_max = jnp.array(iter_max, dtype = 'int32'),
        proposal_iter_sample = jnp.array(0, dtype = 'int32')
      )

    # Define step function
    def step(self, fsm_idx, state):
        """ Take a single step of the FSM """
        fsm_idx, state, sample_taken, sample_loc = jax.lax.switch(
            fsm_idx,
            [
                self.PROPOSE.step,
                self.ACCEPT.step,
                self.UPDATE.step,
                self.DONE.step,
            ],
            state,
        )
    
        return fsm_idx, state, sample_taken, sample_loc