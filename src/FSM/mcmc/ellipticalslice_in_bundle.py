"""
EllipSlice FSM with step bundling
"""

import jax
import jax.numpy as jnp
import jax.random as jrnd
import jax_dataclasses as jdc

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
  ''' Convert the theta to a position. '''
  return state.position * jnp.cos(theta) + state.ellipse * jnp.sin(theta)


class EllipticalSliceFSM:

    def __init__(
        self,
        mean, 
        cov,
        logdensity_fn,
    ):
        EllipticalSliceFSM.mean = mean
        EllipticalSliceFSM.cov = cov
        EllipticalSliceFSM.logdensity_fn = logdensity_fn
    ################################################################################
    class INIT:
      index: jax.Array = jnp.array(0, dtype = jnp.int32)
    
      @classmethod
      def step(cls, state, sample_taken, fsm_idx, sample_loc):
    
        with jdc.copy_and_mutate(state) as new_state:
          thresh_rng, ellipse_rng, bracket_rng, new_state.rng = jrnd.split(state.rng, 4)
    
          # Draw a threshold uniformly.
          u = jrnd.uniform(thresh_rng)
          new_state.thresh = jnp.log(u) + EllipticalSliceFSM.logdensity_fn(state.position)
    
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
          False,                                 # No sample taken.
          EllipticalSliceFSM.SHRINK.index,   # Next state is shrinking ellipse.
          new_state.position                     # sample location
          )
    
    ################################################################################
    class SHRINK:
      index: jax.Array = jnp.array(1, dtype = jnp.int32)
      
      @classmethod
      def step(cls, state, sample_taken, fsm_idx, sample_loc):
        # If the threshold is lower than the logprob, we're done.
        done = EllipticalSliceFSM.logdensity_fn(
            slice_pos(state, state.theta)
        ) > state.thresh
    
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
              False,                                # No sample taken.
              EllipticalSliceFSM.DONE.index,   # Store sample.
              new_state.position
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
          False,                                 # No sample taken.
          cls.index,                             # Next state is this state.
          new_state.position                     # sample location
        )
    
    ################################################################################
    class DONE:
      index: jax.Array =  jnp.array(2, dtype = jnp.int32)
      
      @classmethod
      def step(cls, state, sample_taken, fsm_idx, sample_loc):
        return (
          state,
          True,                                 # Sample taken.
          EllipticalSliceFSM.INIT.index,    # Reset to initial state.
          state.position                     # sample location

        )
    
    
    ################################################################################
    
    # Define init function
    def init(self, rng, init_pos):
      return EllipticalSliceState(
        rng=rng,
        position=init_pos,
        ellipse=jnp.zeros(init_pos.shape),
        thresh=jnp.array(jnp.nan),
        theta=jnp.array(jnp.nan),
        upper=jnp.array(jnp.nan),
        lower=jnp.array(jnp.nan),
      )

    # Define step function
    def step(self, fsm_idx, state):
        """ Take a single step of the FSM """

        sample_taken = False
        sample_loc = state.position

        state, sample_taken, fsm_idx, sample_loc = jax.lax.cond(
            fsm_idx==0,
            self.INIT.step,
            lambda a,b,c,d : (a,b,c,d),
            state, sample_taken, fsm_idx, sample_loc,
        )

        state, sample_taken, fsm_idx, sample_loc = jax.lax.cond(
            fsm_idx==1,
            self.SHRINK.step,
            lambda a,b,c,d : (a,b,c,d),
            state, sample_taken, fsm_idx, sample_loc,
        )

        state, sample_taken, fsm_idx, sample_loc = jax.lax.cond(
            fsm_idx==2,
            self.DONE.step,
            lambda a,b,c,d : (a,b,c,d),
            state, sample_taken, fsm_idx, sample_loc,
        )

        return fsm_idx, state, sample_taken, sample_loc