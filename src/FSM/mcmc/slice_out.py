"""
SliceFSM (amortized logpdf calls).
Runs main `step' until logpdf is required
"""
import jax
import jax.numpy as jnp
import jax.random as jrnd
import jax_dataclasses as jdc


@jdc.pytree_dataclass
class SliceState:
  rng:       jax.Array
  position:   jax.Array
  direction: jax.Array
  thresh:    jax.Array
  alpha:     jax.Array
  upper:     jax.Array
  lower:     jax.Array

def slice_pos(state: SliceState, alpha: jax.Array):
  ''' Convert the alpha to a position. '''
  return state.position + alpha*state.direction

class SliceFSM:

    def __init__(
        self,
        step_size, 
    ):
        SliceFSM.step_size = step_size
 

    ################################################################################
    class INIT:
      index: jax.Array = jnp.array(0, dtype = int)
    
      @classmethod
      def step(cls, state, in_logprob):
    
        with jdc.copy_and_mutate(state) as new_state:
          thresh_rng, dir_rng, bracket_rng, new_state.rng = jrnd.split(state.rng, 4)
    
          # Draw a threshold uniformly.
          u = jrnd.uniform(thresh_rng)
          new_state.thresh = jnp.log(u) + in_logprob
    
          # Draw a direction uniformly.
          direction = jrnd.normal(dir_rng, (state.position.shape[0],))
          new_state.direction = direction / jnp.linalg.norm(direction)
    
          # Randomly center the bracket.
          new_state.upper = jrnd.uniform(
            bracket_rng, 
            minval=0.0, 
            maxval=SliceFSM.step_size,
            )
          new_state.lower = new_state.upper - SliceFSM.step_size
    
        return (
          new_state,
          jnp.nan*jnp.ones_like(state.position), # No logprob to request.
          False,                                # No logprob evaluation is needed here.
          False,                                # No sample taken.
          SliceFSM.UPPER_INIT.index,        # Next state is expanding upper bound.
          )
    
    
    ################################################################################
    class UPPER_INIT:
      index: jax.Array = jnp.array(1, dtype = int)
    
      @classmethod
      def step(cls, state, in_logprob):
        return (
          state,                         # No change to state
          slice_pos(state, state.upper), # Request logprob at upper bound.
          True,                          # New logprob evaluation is needed.
          False,                         # No sample taken.
          SliceFSM.UPPER_LOOP.index, # Next state is expanding upper bound.
          )
    
    
    ################################################################################
    class UPPER_LOOP:
      index: jax.Array = jnp.array(2, dtype = int)
    
      @classmethod
      def _done(cls, state):
        ''' Evaluate when exiting state. '''
        return (
          state,
          jnp.nan*jnp.ones_like(state.position), # No logprob to request.
          False,                                # No logprob evaluation is needed.
          False,                                # No sample taken.
          SliceFSM.LOWER_INIT.index,        # Next state is expanding lower bound.
        )
    
      @classmethod
      def _not_done(cls, state):
        ''' Evaluate when not exiting state. '''
        # Step out the upper bound.
        with jdc.copy_and_mutate(state) as new_state:
          new_state.upper += SliceFSM.step_size
        
        return (
          new_state,
          slice_pos(new_state, new_state.upper), # Request logprob at upper bound.
          True,                                  # New logprob evaluation is needed.
          False,                                 # No sample taken.
          cls.index,                             # Stay in this state.
        )
    
      @classmethod
      def step(cls, state, in_logprob):
        # The threshold needs to be higher than the logprob at the upper bound.
        done = (in_logprob < state.thresh)
    
        return jax.lax.cond(
          done,
          cls._done,
          cls._not_done,
          state,
        )
        
    ################################################################################
    class LOWER_INIT:
      index: jax.Array = jnp.array(3, dtype = int)
    
      @classmethod
      def step(cls, state, in_logprob):
        return (
          state,                         # No change to state
          slice_pos(state, state.lower), # Request logprob at lower bound.
          True,                          # New logprob evaluation is needed.
          False,                         # No sample taken.
          SliceFSM.LOWER_LOOP.index, # Loop in this state.
          )
    
    ################################################################################
    class LOWER_LOOP:
      index: jax.Array = jnp.array(4, dtype = int)
    
      @classmethod
      def _done(cls, state):
        return (
          state,
          jnp.nan*jnp.ones_like(state.position), # No logprob to request.
          False,                                # No logprob evaluation is needed.
          False,                                # No sample taken.
          SliceFSM.SHRINK_INIT.index,       # Next state is shrinking in.
        )
    
      @classmethod
      def _not_done(cls, state):
    
        # Step out the lower bound.
        with jdc.copy_and_mutate(state) as new_state:
          new_state.lower -= SliceFSM.step_size
        
        return (
          new_state,
          slice_pos(new_state, new_state.lower), # Request logprob at lower bound.
          True,                                  # New logprob evaluation is needed.
          False,                                 # No sample taken.
          cls.index,                             # Stay in this state.
        )
    
      @classmethod
      def step(cls, state, in_logprob):
        # The threshold needs to be higher than the logprob at the lower bound.
        done = (in_logprob < state.thresh)
    
        return jax.lax.cond(
          done,
          cls._done,
          cls._not_done,
          state,
        )
        
    ################################################################################
    class SHRINK_INIT:
      index: jax.Array = jnp.array(5, dtype = int)
    
      @classmethod
      def step(cls, state, in_logprob):
    
        # Draw a point uniformly between the upper and lower bounds.
        with jdc.copy_and_mutate(state) as new_state:
          new_alpha_rng, new_state.rng = jrnd.split(state.rng)
          new_state.alpha = jrnd.uniform(
            new_alpha_rng, 
            minval=new_state.lower,
            maxval=new_state.upper,
            )
    
        return (
          new_state,                             # State has new rng.
          slice_pos(new_state, new_state.alpha), # Request logprob at new alpha.
          True,                                  # New logprob evaluation is needed.
          False,                                 # No sample taken.
          SliceFSM.SHRINK_LOOP.index,        # Start shrinking loop.
          )
    
    ################################################################################
    class SHRINK_LOOP:
      index: jax.Array = jnp.array(6, dtype = int)
      
      @classmethod
      def _done(cls, state):
        with jdc.copy_and_mutate(state) as new_state:
          new_state.position = slice_pos(state, state.alpha) # New sample.
    
        return (
          new_state,
          jnp.nan*jnp.ones_like(state.position), # No logprob to request.
          False,                                # No logprob evaluation is needed.
          False,                                # No sample taken.
          SliceFSM.DONE.index,             # Store sample.
        )
    
      @classmethod
      def _not_done(cls, state):
        ''' Evaluate when not exiting state. '''
    
        with jdc.copy_and_mutate(state) as new_state:
    
          # Shrink the bounds.
          new_state.upper = jnp.where(state.alpha > 0, state.alpha, new_state.upper)
          new_state.lower = jnp.where(state.alpha < 0, state.alpha, new_state.lower)
        
          # Get a new rng.
          new_alpha_rng, new_state.rng = jrnd.split(state.rng)
    
          # Draw a new alpha.
          new_state.alpha = jax.random.uniform(
            new_alpha_rng, 
            minval=new_state.lower,
            maxval=new_state.upper,
          )
    
        return (
          new_state,
          slice_pos(new_state, new_state.alpha), # Request logprob at new alpha.
          True,                                  # New logprob evaluation is needed.
          False,                                 # No sample taken.
          cls.index,                             # Next state is this state.
        )
    
      @classmethod
      def step(cls, state, in_logprob):
        # If the threshold is lower than the logprob, we're done.
        done = (in_logprob > state.thresh)
    
        return jax.lax.cond(
          done,
          cls._done,
          cls._not_done,
          state,
        )
    
    ################################################################################
    class DONE:
      index: jax.Array = jnp.array(7, dtype = int)
      
      @classmethod
      def step(cls, state, in_logprob):
        return (
          state,
          state.position,                        # This is the sample location.
          False,                                # No logprob evaluation is needed.
          True,                                 # Sample taken.
          SliceFSM.INIT.index,              # Reset to initial state.
        )
    

################################################################################

    # Define init function
    def init(self, rng, init_pos):
      return SliceState(
        rng=rng,
        position=init_pos,
        direction=jnp.zeros(init_pos.shape),
        thresh=jnp.array(0, dtype = jnp.float32),
        alpha=jnp.array(0, dtype = jnp.float32),
        upper=jnp.array(0, dtype = jnp.float32),
        lower=jnp.array(0, dtype = jnp.float32),

      ) 

    # Defining step function
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
                  self.UPPER_INIT.step,
                  self.UPPER_LOOP.step,
                  self.LOWER_INIT.step,
                  self.LOWER_LOOP.step,
                  self.SHRINK_INIT.step,
                  self.SHRINK_LOOP.step,
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