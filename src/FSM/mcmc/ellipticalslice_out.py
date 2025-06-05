"""
EllipSlice FSM with amortized logpdf
(calls main `step' until logdf required)
"""
import jax
import jax.numpy as jnp
import jax.random as jrnd
import jax_dataclasses as jdc

# Giving jax Array attribute if using 0.3.X
import typing
if not hasattr(jax,"Array"):
    jax.Array = typing.Any

@jdc.pytree_dataclass
class EllipticalSliceState:
  rng:       jax.Array
  position:   jax.Array
  ellipse:   jax.Array
  thresh:    jax.Array
  theta:     jax.Array
  upper:     jax.Array
  lower:     jax.Array

def slice_pos(state: EllipticalSliceState, theta: jax.Array):
    return (
        (state.position - EllipticalSliceFSM.mean) * jnp.cos(theta) +
        (state.ellipse - EllipticalSliceFSM.mean) * jnp.sin(theta) +
        EllipticalSliceFSM.mean
    )

class EllipticalSliceFSM:

    def __init__(
        self,
        mean, 
        cov,
    ):
        EllipticalSliceFSM.mean = mean
        EllipticalSliceFSM.cov = cov
        
    ################################################################################
    class INIT:
      index: jax.Array = jnp.array(0, dtype = int)
    
      @classmethod
      def step(cls, state, in_logprob):
    
        with jdc.copy_and_mutate(state) as new_state:
          thresh_rng, ellipse_rng, bracket_rng, new_state.rng = jrnd.split(state.rng, 4)
    
          # Draw a threshold uniformly.
          u = jrnd.uniform(thresh_rng)
          new_state.thresh = jnp.log(u) + in_logprob
    
          # Draw an ellipse from N(mean,cov).
          new_state.ellipse = jrnd.multivariate_normal(ellipse_rng, 
                                                       EllipticalSliceFSM.mean, 
                                                       EllipticalSliceFSM.cov)
    
          # Randomly draw theta and initialise bracket
          new_state.theta = jrnd.uniform(
            bracket_rng, 
            minval=0.0, 
            maxval=2*jnp.pi,
            )
          new_state.upper = new_state.theta
          new_state.lower = new_state.theta - 2*jnp.pi
    
        return (
          new_state,                             # State has been updated.
          slice_pos(new_state, new_state.theta), # Request logprob at initial theta.
          True,                                  # logprob evaluation is needed.
          False,                                 # No sample taken..
          EllipticalSliceFSM.SHRINK.index,   # Next state is shrinking ellipse.
          )
    
    ################################################################################
    class SHRINK:
        index: jax.Array =  jnp.array(1, dtype = int)
        
        @classmethod
        def step(cls, state, in_logprob):
            # If the threshold is lower than the logprob, we're done.
            done = in_logprob > state.thresh
            
            return jax.lax.cond(
              done,
              cls._done,
              cls._not_done,
              state,
            )
        
        @classmethod
        def _done(cls, state):
            ''' Update current position and exit state '''
            with jdc.copy_and_mutate(state) as new_state:
                new_state.position = slice_pos(state, state.theta) # New sample.
        
            return (
              new_state,
              jnp.nan*jnp.ones_like(state.position), # No logprob to request.
              False,                                # No logprob evaluation is needed.
              False,                                # No sample taken.
              EllipticalSliceFSM.DONE.index,    # Store sample.
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
              slice_pos(new_state, new_state.theta), # Request logprob at new theta.
              True,                                  # New logprob evaluation is needed.
              False,                                 # No sample taken.
              cls.index,                             # Next state is this state.
            )
    
    ################################################################################
    class DONE:
        index: jax.Array =  jnp.array(2, dtype = int)
        
        @classmethod
        def step(cls, state, in_logprob):
            return (
              state,
              state.position,                        # This is the sample location.
              False,                                # No logprob evaluation is needed.
              True,                                 # Sample taken.
              EllipticalSliceFSM.INIT.index,    # Reset to initial state.
            )

    
    ################################################################################

    # Define init function
    def init(self, rng, init_pos):
      return EllipticalSliceState(
        rng=rng,
        position=init_pos,
        ellipse=jnp.zeros(init_pos.shape),
        thresh=jnp.array(0, dtype = jnp.float32),
        theta=jnp.array(0, dtype = jnp.float32),
        upper=jnp.array(0, dtype = jnp.float32),
        lower=jnp.array(0, dtype = jnp.float32),
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
        
            sample_loc = jnp.where(take_sample, req_pos, sample_loc)
            return fsm_idx, state, req_pos, eval_logprob, have_sample | take_sample, sample_loc
    
        # Step until condition satisfied
        init_loop_state = (
                           fsm_idx, 
                           state, 
                           jnp.nan * jnp.ones_like(state.position), 
                           False, 
                           False, 
                           jnp.nan * jnp.ones_like(state.position)
                           )
        
        loop_state = jax.lax.while_loop(_step_cond, _step_body, init_loop_state)
        fsm_idx, state, req_pos, eval_logprob, sample_taken, sample_loc = loop_state
        
        return fsm_idx, state, req_pos, sample_taken, sample_loc