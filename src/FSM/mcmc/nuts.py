"""
NUTS FSM basic (uses BlackJax helpers)
"""
import jax
import jax.numpy as jnp
import jax.random as jrnd
import jax_dataclasses as jdc
from functools import partial

import blackjax.mcmc.hmc as hmc
import blackjax.mcmc.integrators as integrators
import blackjax.mcmc.metrics as metrics
import blackjax.mcmc.proposal as proposal
import blackjax.mcmc.trajectory as trajectory
import blackjax.mcmc.termination as termination

# Global functions
progressive_uniform_sampling = proposal.progressive_uniform_sampling
progressive_biased_sampling = proposal.progressive_biased_sampling
proposal_generator = proposal.proposal_generator
velocity_verlet = integrators.velocity_verlet
hmc_energy = trajectory.hmc_energy
sample_proposal = progressive_uniform_sampling
proposal_sampler = progressive_biased_sampling

@jdc.pytree_dataclass
class NUTSState:
    # Core HMC Variables
    hmc_state: hmc.HMCState
    momentum: jax.Array

    # Symplectic Integration Variables
    global_trajectory_state: trajectory.DynamicExpansionState
    local_trajectory_state: trajectory.DynamicIntegrationState
    direction: jax.Array

    # Termination and Trajectory Management
    termination_state: termination.IterativeUTurnState
    is_turning: jax.Array
    is_diverging: jax.Array

    # Energy Variables
    energy: jax.Array

    # RNG Management
    rng: jax.Array

class NutsFSM:
    def __init__(
        self,
        step_size,
        max_num_expansions,
        divergence_threshold,
        inverse_mass_matrix,
        logdensity_fn,
    ):
        NutsFSM.step_size = step_size
        NutsFSM.max_num_expansions = max_num_expansions
        NutsFSM.divergence_threshold = divergence_threshold
        NutsFSM.inverse_mass_matrix = inverse_mass_matrix
        NutsFSM.logdensity_fn = logdensity_fn
        NutsFSM.metric = metrics.default_metric(inverse_mass_matrix)
        (
        NutsFSM.new_termination_state,
        NutsFSM.update_termination_state,
        NutsFSM.is_criterion_met,
        ) = termination.iterative_uturn_numpyro(NutsFSM.metric.check_turning) 
        _, NutsFSM.generate_proposal = proposal_generator(hmc_energy(NutsFSM.metric.kinetic_energy))
        NutsFSM.integrator = velocity_verlet(NutsFSM.logdensity_fn,
                                             NutsFSM.metric.kinetic_energy)

    ################################################################################
    class INIT:
        index: int = 0

        @classmethod
        def step(cls, state):
            """Sample momentum and initialise global trajectory"""
            with jdc.copy_and_mutate(state) as new_state:
                
                # Sample momentum
                key_momentum, new_state.rng = jax.random.split(state.rng, 2)
                position, logdensity, logdensity_grad = state.hmc_state
                new_state.momentum = NutsFSM.metric.sample_momentum(key_momentum, position)
    
                #################################################################################
                # Initialisation for global trajectory
                
                # initialising integrator state (pos, mom, logdensity, logdensity_grad)
                position, logdensity, logdensity_grad = state.hmc_state
                initial_integrator_state = integrators.IntegratorState(
                position, new_state.momentum, logdensity, logdensity_grad
                )
    
                # initialising energy at current state
                initial_energy = (
                            -initial_integrator_state.logdensity
                            + NutsFSM.metric.kinetic_energy(initial_integrator_state.momentum)
                         )
    
                # initialising termination state for integration (class IterativeUTurnState)    
                initial_termination_state = NutsFSM.new_termination_state(
                    initial_integrator_state, NutsFSM.max_num_expansions
                )
    
                # initialising proposal at starting point
                initial_proposal = proposal.Proposal(
                    initial_integrator_state, 
                    initial_energy, 
                    jnp.array(0, dtype=jnp.float32), 
                    -jnp.array(jnp.inf,dtype = jnp.float32)
                )
    
                # initialising global trajectory as singleton around initial_state
                initial_global_trajectory = trajectory.Trajectory(
                    leftmost_state = initial_integrator_state,
                    rightmost_state = initial_integrator_state,
                    momentum_sum = initial_integrator_state.momentum,
                    num_states = jnp.array(0, dtype=jnp.int32),
                )
    
                # Storing in global state (step, proposal, traj, term_state)
                initial_expansion_state = trajectory.DynamicExpansionState(
                    jnp.array(0, dtype=jnp.int32), 
                    initial_proposal, 
                    initial_global_trajectory, 
                    initial_termination_state
                )
                #################################################################################
    
                # Save into new_state
                new_state.energy = initial_energy
                new_state.global_trajectory_state = initial_expansion_state

            return new_state, False, NutsFSM.EXPAND.index, new_state.hmc_state.position

    ################################################################################
    class EXPAND:
        index: int = 1

        @classmethod
        def step(cls, state: NUTSState):
            """Expand global trajectory and set up local_trajectory_state"""
            with jdc.copy_and_mutate(state) as new_state:
                
                # Extract relevant global state information
                (
                    step, 
                    global_trajectory, 
                    termination_state
                ) = (
                    state.global_trajectory_state.step, 
                    state.global_trajectory_state.trajectory, 
                    state.global_trajectory_state.termination_state
                )
                
                # pick direction to expand along
                subkey = jax.random.fold_in(state.rng, step)
                direction_key, proposal_key = jax.random.split(subkey, 2)
                direction = jnp.where(jax.random.bernoulli(direction_key), 1, -1)
    
                #################################################################################
                # Initialisation for local trajectory
    
                # choose starting point for local trajectory
                start_state = jax.lax.cond(
                    direction > 0,
                    lambda _: global_trajectory.rightmost_state,
                    lambda _: global_trajectory.leftmost_state,
                    operand=None,
                )   
    
                # initialise local proposal at start point
                initial_proposal = NutsFSM.generate_proposal(state.energy, start_state)
    
                # initialise local trajectory as a singleton at start point
                initial_local_trajectory = trajectory.Trajectory(
                    leftmost_state = start_state,
                    rightmost_state = start_state,
                    momentum_sum = start_state.momentum,
                    num_states = jnp.array(0,dtype=jnp.int32),
                )
    
                # Store in local state (step, proposal, traj, term_state)
                initial_integration_state = trajectory.DynamicIntegrationState(
                    jnp.array(0,dtype = jnp.int32), initial_proposal, initial_local_trajectory, termination_state,
                )
                
                #################################################################################
    
                # Save into new_state
                new_state.local_trajectory_state = initial_integration_state
                new_state.direction = direction
            return new_state, False, NutsFSM.INTEGRATE.index, new_state.hmc_state.position

    ################################################################################
    class INTEGRATE:
        index: int = 2

    
        @classmethod
        def step(cls, state: NUTSState):
            """Perform a single integration step and decide whether to continue integrating."""
            with jdc.copy_and_mutate(state) as new_state:
    
                # Unpacking relevant state information
                step, proposal, local_trajectory, termination_state = state.local_trajectory_state
    
                # New key (check it gets used)
                proposal_key, new_state.rng = jax.random.split(state.rng, 2)
    
                # Perform one step of integration - currently redefines the integrator at every step
                new_integrator_state = NutsFSM.integrator(local_trajectory.rightmost_state, 
                                                  state.direction * NutsFSM.step_size)
    
                # Compute new proposal
                new_proposal = NutsFSM.generate_proposal(state.energy, new_integrator_state)
    
                # Check for divergence
                is_diverging = -new_proposal.weight > NutsFSM.divergence_threshold
        
                # Update local trajectory
                # At step 0, we always accept the proposal, since we
                # take one step to get onto the new trajectory
                (new_trajectory, sampled_proposal) = jax.lax.cond(
                    step == 0,
                    lambda _: (
                        trajectory.Trajectory(new_integrator_state, 
                                   new_integrator_state, 
                                   new_integrator_state.momentum, 
                                   jnp.array(1,dtype = jnp.int32)),
                        new_proposal,
                    ),
                    lambda _: (
                        trajectory.append_to_trajectory(local_trajectory, new_integrator_state),
                        sample_proposal(proposal_key, proposal, new_proposal),
                    ),
                    operand=None,
                ) 
    
                # Update termination_state
                new_termination_state = NutsFSM.update_termination_state(
                    termination_state, new_trajectory.momentum_sum, new_integrator_state.momentum, step
                )
    
                # Check if terminated based on U-turn check fn
                has_terminated = NutsFSM.is_criterion_met(
                    new_termination_state, new_trajectory.momentum_sum, new_integrator_state.momentum
                )
    
                # Update local state
                new_local_trajectory_state = trajectory.DynamicIntegrationState(
                    step + 1, sampled_proposal, new_trajectory,new_termination_state,
                )
    
                # Save into new_state
                new_state.local_trajectory_state = new_local_trajectory_state
                new_state.is_turning = has_terminated
                new_state.is_diverging = is_diverging
    
            # Exit state if diverging, turning or max-num-steps
            done = (new_state.is_turning | 
                    new_state.is_diverging | 
                    (new_state.local_trajectory_state.step >=2**new_state.global_trajectory_state.step))
            return jax.lax.cond(
                                  done,
                                  cls._done,
                                  cls._not_done,
                                  new_state,
                                )

        @classmethod
        def _done(cls, state):
            return state, False, NutsFSM.CHECK.index, state.hmc_state.position

        @classmethod
        def _not_done(cls, state):
            return state, False, cls.index, state.hmc_state.position

    ################################################################################
    class CHECK:
        index: int = 3
    
        @classmethod
        def step(cls, state: NUTSState):
            """Handle trajectory expansion logic and decide whether to continue doubling."""
            with jdc.copy_and_mutate(state) as new_state:
    
                # New key (check it gets used)
                proposal_key, new_state.rng = jax.random.split(state.rng, 2)
    
    
                # Unpacking relevant state information
                local_step, local_proposal, local_trajectory, local_termination_state = state.local_trajectory_state
                global_step, global_proposal, global_trajectory, global_termination_state = state.global_trajectory_state
                direction = state.direction
                
                # Swap order of trajectory if direction = -1 (integration adds onto rightmoststate)
                local_trajectory = jax.lax.cond(
                    direction > 0,
                    lambda _: local_trajectory,
                    lambda _: trajectory.Trajectory(
                        local_trajectory.rightmost_state,
                        local_trajectory.leftmost_state,
                        local_trajectory.momentum_sum,
                        local_trajectory.num_states,
                    ),
                    operand=None,
                )
    
                # Update the proposal
                #
                # We do not accept proposals that come from diverging or turning
                # subtrajectories. However the definition of the acceptance probability is
                # such that the acceptance probability needs to be computed across the
                # entire trajectory.
                def update_sum_log_p_accept(inputs):
                    _, cur_proposal, new_proposal = inputs
                    return proposal.Proposal(
                        cur_proposal.state,
                        cur_proposal.energy,
                        cur_proposal.weight,
                        jnp.logaddexp(
                            cur_proposal.sum_log_p_accept, new_proposal.sum_log_p_accept
                        ),
                    )
    
                updated_proposal = jax.lax.cond(
                    state.is_diverging | state.is_turning,
                    update_sum_log_p_accept,
                    lambda x: proposal_sampler(*x),
                    operand=(proposal_key, global_proposal, local_proposal),
                )
    
                # Is the full trajectory making a U-Turn?
                #
                # We first merge the subtrajectory that was just generated with the
                # trajectory and check the U-Turn criteria on the whole trajectory.
                left_trajectory, right_trajectory = trajectory.reorder_trajectories(
                    direction, global_trajectory, local_trajectory
                )
                merged_trajectory = trajectory.merge_trajectories(left_trajectory, right_trajectory)
    
                is_turning_global = NutsFSM.metric.check_turning(
                    merged_trajectory.leftmost_state.momentum,
                    merged_trajectory.rightmost_state.momentum,
                    merged_trajectory.momentum_sum,
                )
    
                # Update global state
                new_global_trajectory_state = trajectory.DynamicExpansionState(
                    global_step + 1, updated_proposal, merged_trajectory, local_termination_state
                )
    
                # Storing new results
                new_state.global_trajectory_state = new_global_trajectory_state
                new_state.is_turning = is_turning_global
                
            # Exit state if diverging, turning or max-num-steps
            done = (new_state.is_diverging |
                    new_state.is_turning |
                    (new_state.global_trajectory_state.step >= NutsFSM.max_num_expansions))
            
            return jax.lax.cond(
                                  done,
                                  cls._done,
                                  cls._not_done,
                                  new_state,
                                )
        @classmethod
        def _done(cls, state):
            ''' Evaluate when exiting state. '''
            return state, False, NutsFSM.DONE.index, state.hmc_state.position
        
        
        @classmethod
        def _not_done(cls, state):
            ''' Evaluate when not exiting state. '''
            return state, False, NutsFSM.EXPAND.index, state.hmc_state.position
    ################################################################################
    class DONE:
        index: int = 4

        @classmethod
        def step(cls, state: NUTSState):
            """Finalize the NUTS trajectory and produce a sample."""
            with jdc.copy_and_mutate(state) as new_state:
    
                # Extracting state information
                num_doublings, sampled_proposal, global_trajectory, _ = new_state.global_trajectory_state
                proposal = sampled_proposal.state
                
                # Getting final HMC state
                hmc_state = hmc.HMCState(
                                            proposal.position,
                                            proposal.logdensity,
                                            proposal.logdensity_grad
                                        )
                    
                # Storing new results
                new_state.hmc_state = hmc_state

            return new_state, True, NutsFSM.INIT.index, new_state.hmc_state.position

    ################################################################################

    def step(self, fsm_idx, state):
        """ Take a single step of the FSM """
        state, sample_taken, fsm_idx, sample_loc = jax.lax.switch(
            fsm_idx,
            [
                self.INIT.step,
                self.EXPAND.step,
                self.INTEGRATE.step,
                self.CHECK.step,
                self.DONE.step,
            ],
            state,
        )
        
        return fsm_idx, state, sample_taken, sample_loc
        
    def init(self, rng, init_pos):
        """Initialize the NUTS state for multiple chains."""
        metric = NutsFSM.metric
    
        # Split RNG for initialization
        key_momentum, rng = jax.random.split(rng)
    
        # Evaluate the logprob and gradient at the initial position
        logdensity, logdensity_grad = jax.value_and_grad(NutsFSM.logdensity_fn)(init_pos)
    
        # Sample initial momentum
        initial_momentum = metric.sample_momentum(key_momentum, init_pos)
    
        # Initialize HMC state
        hmc_state = hmc.HMCState(
            position=init_pos,
            logdensity=logdensity,
            logdensity_grad=logdensity_grad,
        )
    
        # Initialize integrator state
        initial_integrator_state = integrators.IntegratorState(
            position=init_pos,
            momentum=initial_momentum,
            logdensity=logdensity,
            logdensity_grad=logdensity_grad,
        )
    
        # Initialise energy
        initial_energy = -logdensity + metric.kinetic_energy(initial_momentum)
    
        termination_state = NutsFSM.new_termination_state(
            initial_integrator_state, NutsFSM.max_num_expansions
        )

        # Initialize global trajectory
        global_trajectory_state = trajectory.DynamicExpansionState(
            step=jnp.array(0,dtype=jnp.int32),
            proposal=proposal.Proposal(
                state=initial_integrator_state, energy=initial_energy, weight=0.0, sum_log_p_accept=-jnp.inf
            ),
            trajectory=trajectory.Trajectory(
                leftmost_state=initial_integrator_state,
                rightmost_state=initial_integrator_state,
                momentum_sum=initial_momentum,
                num_states=jnp.array(0,dtype=jnp.int32),
            ),
            termination_state=termination_state,
        )
    
        # Initialize local trajectory
        local_trajectory_state = trajectory.DynamicIntegrationState(
            step=jnp.array(0,dtype=jnp.int32),
            proposal=proposal.Proposal(
                state=initial_integrator_state, energy=-jnp.inf, weight=0.0, sum_log_p_accept=-jnp.inf
            ),
            trajectory=trajectory.Trajectory(
                leftmost_state=initial_integrator_state,
                rightmost_state=initial_integrator_state,
                momentum_sum=jnp.zeros_like(init_pos),
                num_states=jnp.array(0,dtype=jnp.int32),
            ),
            termination_state=termination_state,
        )
    
        # Initialize chain-specific variables
        is_turning = False
        is_diverging = False
        direction = 1  # Default integration direction; updated during trajectory expansion
    
        # Return the NUTSState object
        return NUTSState(
            hmc_state=hmc_state,
            momentum=initial_momentum,
            global_trajectory_state=global_trajectory_state,
            local_trajectory_state=local_trajectory_state,
            direction=direction,
            termination_state=termination_state,
            is_turning=is_turning,
            is_diverging=is_diverging,
            energy=initial_energy,
            rng=rng,
        )