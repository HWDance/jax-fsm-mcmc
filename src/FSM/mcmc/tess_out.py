"""
TransportEllipSliceFSM with amortized logpdf calls
runs main `step' until logpdf call is needed.
"""
import jax
import jax.numpy as jnp
import jax.random as jrnd
import jax_dataclasses as jdc
from jax.flatten_util import ravel_pytree


# Giving jax Array attribute if using 0.3.X
import typing
if not hasattr(jax,"Array"):
    jax.Array = typing.Any

@jdc.pytree_dataclass
class TessState:
  rng:          jax.Array
  position_u:   jax.Array
  position_x:   jax.Array
  position_v:   jax.Array
  thresh:       jax.Array
  theta:        jax.Array
  upper:        jax.Array
  lower:        jax.Array

def slice_pos_u(state: TessState, theta: jax.Array):
    """
    Update positions in the flattened arrays within dictionaries.
    """
    # Flatten position_u and position_v dictionaries
    flat_u, unravel_u = ravel_pytree(state.position_u)
    flat_v, unravel_v = ravel_pytree(state.position_v)

    # Perform operations on flattened arrays
    updated_u = (
        flat_u * jnp.cos(theta) 
        + flat_v * jnp.sin(theta) 
    )
    
    return unravel_u(updated_u)

def slice_pos(state: TessState, theta: jax.Array):
    """
    Update positions in the flattened arrays within dictionaries.
    """
    # Flatten position_u and position_v dictionaries
    flat_u, unravel_u = ravel_pytree(state.position_u)
    flat_v, unravel_v = ravel_pytree(state.position_v)

    # Perform operations on flattened arrays
    updated_u = (
        flat_u * jnp.cos(theta) 
        + flat_v * jnp.sin(theta) 
    )
    updated_v = (
        flat_v * jnp.cos(theta) 
        - flat_u * jnp.sin(theta) 
    )

    return unravel_u(updated_u), unravel_v(updated_v)

def momentum_generator(rng_key, position):
    flat_position, unravel = ravel_pytree(position)
    return unravel(jax.random.normal(rng_key, shape=jnp.shape(flat_position)))

class TessFSM:

    def __init__(
        self,
        mean=0.0, 
        cov=1.0,
    ):
        TessFSM.mean = mean
        TessFSM.cov = cov
        
    ################################################################################
    class INIT:
      index: jax.Array = jnp.array(0, dtype = int)
    
      @classmethod
      def step(cls, state, in_logprob):
    
        with jdc.copy_and_mutate(state) as new_state:
            ellipse_rng, thresh_rng, bracket_rng, new_state.rng = jrnd.split(state.rng, 4)
            
            # Recovering logprob and state position
            logprob, _ = in_logprob

            # Update ellipse (position_v) 
            new_state.position_v = momentum_generator(
                ellipse_rng, 
                new_state.position_u
            )
            
            # Draw a threshold uniformly.
            w = jrnd.uniform(thresh_rng)
            new_state.thresh = logprob + jnp.log(w)

            # Randomly draw theta and initialise bracket
            new_state.theta = jrnd.uniform(
                bracket_rng, 
                minval=0.0, 
                maxval=2*jnp.pi,
            )
            new_state.upper = new_state.theta
            new_state.lower = new_state.theta - 2*jnp.pi
    
        return (
          new_state,                                   # State has been updated.
          slice_pos(new_state, new_state.theta),       # Request logprob at new theta.
          True,                                        # logprob evaluation is needed.
          False,                                       # No sample taken..
          TessFSM.SHRINK.index,                        # Next state is shrinking ellipse.
          )
    
    ################################################################################
    class SHRINK:
        index: jax.Array =  jnp.array(1, dtype = int)
        
        @classmethod
        def step(cls, state, in_logprob):
            # If the threshold is lower than the logprob, we're done.
            logprob, _ = in_logprob
            done = logprob > state.thresh 
            
            return jax.lax.cond(
              done,
              cls._done,
              cls._not_done,
              state,
            )
        
        @classmethod
        def _done(cls, state):
            ''' exit state '''
            return (
              state,
              slice_pos(state, state.theta), # Request logprob at new theta.
              False,                                 # No logprob evaluation is needed.
              False,                                 # No sample taken.
              TessFSM.DONE.index,    # Store sample.
                )
            
        @classmethod
        def _not_done(cls, state):
            ''' Shrink ellipse and continue '''
            with jdc.copy_and_mutate(state) as new_state: 
                
                # Shrink the bounds.
                new_state.upper = jnp.where(state.theta > 0, state.theta, new_state.upper)
                new_state.lower = jnp.where(state.theta < 0, state.theta, new_state.lower)
            
                # Get a new rng.
                new_theta_rng, new_state.rng = jrnd.split(state.rng)
                
                # Draw a new theta.
                new_state.theta = jax.random.uniform(
                    new_theta_rng, 
                    minval=new_state.lower,
                    maxval=new_state.upper,
                )

                
            return (
              new_state,
              slice_pos(new_state, new_state.theta),       # Request logprob at new theta.
              True,                                        # New logprob evaluation is needed.
              False,                                       # No sample taken.
              cls.index,                                   # Next state is this state.
            )
    
    ################################################################################
    class DONE:
        index: jax.Array =  jnp.array(2, dtype = int)
        
        @classmethod
        def step(cls, state, in_logprob):

            with jdc.copy_and_mutate(state) as new_state:

                # Update sample
                _, new_state.position_x = in_logprob
                new_state.position_u = slice_pos_u(state, state.theta)
        
            return (
              new_state,
              (new_state.position_u, new_state.position_v), # Return sample (T(u),v).
              False,                                        # No logprob evaluation is needed.
              True,                                         # Sample taken.
              TessFSM.INIT.index,                           # Reset to initial state.
            )

    ################################################################################

    # Define init function
    def init(self, rng, init_pos):

        # Unpacking latent positions and init sample
        position_u, position_v, position_x = init_pos  
        
        return TessState(
            rng=rng,
            position_u=position_u,
            position_v=position_v,
            position_x=position_x,
            thresh=jnp.array(0, dtype = jnp.float64),
            theta=jnp.array(0, dtype = jnp.float64),
            upper=jnp.array(0, dtype = jnp.float64),
            lower=jnp.array(0, dtype = jnp.float64),
        )    
    
    # Define step function step function
    def step(self, fsm_idx, state, in_logprob):
      
        # Condition to keep iterating
        def _step_cond(loop_state):
            fsm_idx, state, req_pos, eval_logprob, have_sample, sample_loc = loop_state
            return jnp.logical_not(eval_logprob)
    
        # Single fsm iteration   
        def _step_body(loop_state):
            fsm_idx, state, req_pos, eval_logprob, have_sample, sample_loc = loop_state
            state, req_pos, eval_logprob, take_sample, fsm_idx = jax.lax.switch(
              fsm_idx,
              [
                self.INIT.step,
                self.SHRINK.step,
                self.DONE.step,
            ],
              state, in_logprob,
            )

            req_pos_x, req_pos_v = req_pos
            flat_x, unflatten_x = ravel_pytree(req_pos_x)
            flat_loc, unflatten_loc = ravel_pytree(sample_loc)
            sample_loc = unflatten_loc(jnp.where(take_sample, flat_x, flat_loc))
            return fsm_idx, state, req_pos, eval_logprob, have_sample | take_sample, sample_loc
    
        # Step until condition satisfied
        init_loop_state = (
                           fsm_idx, 
                           state, 
                           (state.position_u, state.position_v),
                           False, 
                           False, 
                           state.position_x, 
                           )
        
        loop_state = jax.lax.while_loop(_step_cond, _step_body, init_loop_state)
        fsm_idx, state, req_pos, eval_logprob, sample_taken, sample_loc = loop_state
        
        return fsm_idx, state, req_pos, sample_taken, sample_loc