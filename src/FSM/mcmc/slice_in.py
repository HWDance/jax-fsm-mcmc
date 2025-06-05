"""
SliceFSM (basic)
"""
import jax
import jax.numpy as jnp
import jax.random as jrnd
import jax_dataclasses as jdc
from FSM.base.fsm_base import get_fsm_functions


@jdc.pytree_dataclass
class SliceSamplerState:
  rng:       jax.Array
  cur_pos:   jax.Array
  direction: jax.Array
  thresh:    jax.Array
  alpha:     jax.Array
  upper:     jax.Array
  lower:     jax.Array
  step_sz:   jax.Array

def slice_pos(state: SliceSamplerState, alpha: jax.Array):
  ''' Convert the alpha to a position. '''
  return state.cur_pos + alpha*state.direction

################################################################################
class SliceSampler_INIT:
  index: jax.Array

  @classmethod
  def step(cls, state, in_logprob):

    with jdc.copy_and_mutate(state) as new_state:
      thresh_rng, dir_rng, bracket_rng, new_state.rng = jrnd.split(state.rng, 4)

      # Draw a threshold uniformly.
      u = jrnd.uniform(thresh_rng)
      new_state.thresh = jnp.log(u) + in_logprob

      # Draw a direction uniformly.
      dir = jrnd.normal(dir_rng, (state.cur_pos.shape[0],))
      new_state.direction = dir / jnp.linalg.norm(dir)

      # Randomly center the bracket.
      new_state.upper = jrnd.uniform(
        bracket_rng, 
        minval=0.0, 
        maxval=state.step_sz,
        )
      new_state.lower = new_state.upper - state.step_sz

    return (
      new_state,
      jnp.nan*jnp.ones_like(state.cur_pos), # No logprob to request.
      False,                                # No logprob evaluation is needed here.
      False,                                # No sample taken.
      SliceSampler_UPPER_INIT.index,        # Next state is expanding upper bound.
      )


################################################################################
class SliceSampler_UPPER_INIT:
  index: jax.Array

  @classmethod
  def step(cls, state, in_logprob):
    return (
      state,                         # No change to state
      slice_pos(state, state.upper), # Request logprob at upper bound.
      True,                          # New logprob evaluation is needed.
      False,                         # No sample taken.
      SliceSampler_UPPER_LOOP.index, # Next state is expanding upper bound.
      )


################################################################################
class SliceSampler_UPPER_LOOP:
  index: jax.Array

  @classmethod
  def _done(cls, state):
    ''' Evaluate when exiting state. '''
    return (
      state,
      jnp.nan*jnp.ones_like(state.cur_pos), # No logprob to request.
      False,                                # No logprob evaluation is needed.
      False,                                # No sample taken.
      SliceSampler_LOWER_INIT.index,        # Next state is expanding lower bound.
    )

  @classmethod
  def _not_done(cls, state):
    ''' Evaluate when not exiting state. '''
    # Step out the upper bound.
    with jdc.copy_and_mutate(state) as new_state:
      new_state.upper += state.step_sz
    
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
    done = in_logprob < state.thresh

    return jax.lax.cond(
      done,
      cls._done,
      cls._not_done,
      state,
    )
    
################################################################################
class SliceSampler_LOWER_INIT:
  index: jax.Array

  @classmethod
  def step(cls, state, in_logprob):
    return (
      state,                         # No change to state
      slice_pos(state, state.lower), # Request logprob at lower bound.
      True,                          # New logprob evaluation is needed.
      False,                         # No sample taken.
      SliceSampler_LOWER_LOOP.index, # Loop in this state.
      )

################################################################################
class SliceSampler_LOWER_LOOP:
  index: jax.Array

  @classmethod
  def _done(cls, state):
    return (
      state,
      jnp.nan*jnp.ones_like(state.cur_pos), # No logprob to request.
      False,                                # No logprob evaluation is needed.
      False,                                # No sample taken.
      SliceSampler_SHRINK_INIT.index,       # Next state is shrinking in.
    )

  @classmethod
  def _not_done(cls, state):

    # Step out the lower bound.
    with jdc.copy_and_mutate(state) as new_state:
      new_state.lower -= state.step_sz
    
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
    done = in_logprob < state.thresh

    return jax.lax.cond(
      done,
      cls._done,
      cls._not_done,
      state,
    )
    
################################################################################
class SliceSampler_SHRINK_INIT:
  index: jax.Array

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
      SliceSampler_SHRINK_LOOP.index,        # Start shrinking loop.
      )

################################################################################
class SliceSampler_SHRINK_LOOP:
  index: jax.Array
  
  @classmethod
  def _done(cls, state):
    with jdc.copy_and_mutate(state) as new_state:
      new_state.cur_pos = slice_pos(state, state.alpha) # New sample.

    return (
      new_state,
      jnp.nan*jnp.ones_like(state.cur_pos), # No logprob to request.
      False,                                # No logprob evaluation is needed.
      False,                                # No sample taken.
      SliceSampler_FINAL.index,             # Store sample.
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
    done = in_logprob > state.thresh

    return jax.lax.cond(
      done,
      cls._done,
      cls._not_done,
      state,
    )

################################################################################
class SliceSampler_FINAL:
  index: jax.Array
  
  @classmethod
  def step(cls, state, in_logprob):
    return (
      state,
      state.cur_pos,                        # This is the sample location.
      False,                                # No logprob evaluation is needed.
      True,                                 # Sample taken.
      SliceSampler_INIT.index,              # Reset to initial state.
    )


################################################################################
SliceSamplerClasses = [
  SliceSampler_INIT,
  SliceSampler_UPPER_INIT,
  SliceSampler_UPPER_LOOP,
  SliceSampler_LOWER_INIT,
  SliceSampler_LOWER_LOOP,
  SliceSampler_SHRINK_INIT,
  SliceSampler_SHRINK_LOOP,
  SliceSampler_FINAL,
]

# Assign indices to each state class.
for index, cls in enumerate(SliceSamplerClasses):
  cls.index = index

# Define init function
def init_sampler_state(rng, init_pos, step_sz = 1.0):
  return SliceSamplerState(
    rng=rng,
    cur_pos=init_pos,
    direction=jnp.zeros(init_pos.shape),
    thresh=jnp.array(jnp.nan),
    alpha=jnp.array(jnp.nan),
    upper=jnp.array(jnp.nan),
    lower=jnp.array(jnp.nan),
    step_sz=jnp.array(step_sz),
  ) 

# Get fsm step and update functions
step, get_fsm_updates, get_fsm_samples = get_fsm_functions(SliceSamplerClasses)