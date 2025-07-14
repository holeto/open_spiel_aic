import jax
import os
import chex
import optax


import numpy as np
import jax.numpy as jnp
import orbax.checkpoint as ocp

from flax import nnx
from functools import partial

from open_spiel.python.algorithms.mu_zero.vq_vae_test.point_card_matching import PointCardMatching, PointCardMatchingState
from open_spiel.python.algorithms.mu_zero.vq_vae_test.vq_vae_networks import Afterstate_representation_function, Afterstate_dynamics_function, Afterstate_decoder_function, Policy_function
from open_spiel.python.algorithms.mu_zero.vq_vae_test.train_utils import get_reference_policy, check_param_difference




@chex.dataclass(frozen=True)
class VQ_VAEConfig():


  batch_size: int = 32
  trajectory_max: int = 2

  #Not needed yet
  #c_state_vtrace: float = 1.0
  #rho_state_vtrace: float = np.inf

  beta_commitment: float = 0.25

  afterstate_dimension: int = 32

  afterstate_decoder_hidden_size: int = 64
  afterstate_representation_hidden_size: int = 64
  policy_hidden_size: int = 64
  afterstate_dynamics_hidden_size: int = 64

  learning_rate: float = 3e-4
  networks_seed: int = 99
  gameplay_seed: int = 42

@chex.dataclass(frozen=True)
class TimeStep():
  
  valid: chex.Array = () # [..., 1]
  state: chex.Array = () # [..., state_dim]
  legal: chex.Array = () # [..., Player, A]
  
  action: chex.Array = () # [..., Player, A]
  #policy: chex.Array = () # [..., Player, A]
  
  #reward: chex.Array = () # [..., 1] Reward after playing an action
  
@chex.dataclass(frozen=True)
class SampleTrajectoryCarry:
  game_state: PointCardMatchingState
  terminal: bool
  legal_actions: chex.Array





class VQ_VAETrainStep(nnx.Module):
  """This class encapsulates the logic needed for a single update.
  Inheriting from nnx.Module to simplify parameter handling as static objects"""

  def __init__(self, config:VQ_VAEConfig, action_dimension, state_dimension, optimizer):
    
    networks_key = jax.random.key(config.networks_seed)
    networks_key, decoder_rngs = self.get_next_network_rngs(networks_key)
    self.afterstate_decoder = Afterstate_decoder_function(config.afterstate_dimension, state_dimension, config.afterstate_decoder_hidden_size, rngs=decoder_rngs)
    networks_key, dynamics_rngs = self.get_next_network_rngs(networks_key)
    self.afterstate_dynamics = Afterstate_dynamics_function(config.afterstate_dimension, action_dimension, config.afterstate_dynamics_hidden_size, rngs=dynamics_rngs)   
    networks_key, representation_rngs = self.get_next_network_rngs(networks_key)
    self.afterstate_representation = Afterstate_representation_function(state_dimension, config.afterstate_dimension, config.afterstate_representation_hidden_size, rngs=representation_rngs)
    networks_key, policy_rngs = self.get_next_network_rngs(networks_key)
    self.policy = Policy_function(config.afterstate_dimension, action_dimension, config.policy_hidden_size, rngs=policy_rngs)

    self.optimizer = nnx.Optimizer(self, optimizer)

    self.beta_commitment = config.beta_commitment


  
  def get_next_network_rngs(self, networks_key):
    networks_key, temp_key = jax.random.split(networks_key)
    return networks_key, nnx.Rngs(temp_key) 

  def get_representation(self, state):
    afterstate = self.afterstate_representation(state)
    return afterstate
  
  def decoder(self, afterstate):
    state = self.afterstate_decoder(afterstate)
    return state
  
  def get_policy_outcome(self, afterstate):
    policy_logits, outcome = self.policy(afterstate)
    return policy_logits, outcome
  
  def get_next_afterstate(self, afterstate, outcome):
    next_afterstate = self.afterstate_dynamics(afterstate, outcome)
    return next_afterstate
  
  @nnx.jit
  def _jit_get_representation(self, state):
    return self.get_representation(state)
  
  @nnx.jit
  def _jit_decoder(self, afterstate):
    return self.decoder(afterstate)
  
  @nnx.jit
  def _jit_get_policy_outcome(self, afterstate):
    return self.get_policy_outcome(afterstate)
  
  @nnx.jit
  def _jit_get_next_afterstate(self, afterstate, outcome):
    return self.get_next_afterstate(afterstate, outcome)
  
  #FOR PROCESSING BATCHES:
  @nnx.jit
  @nnx.vmap(in_axes=(None, 0))
  def vectorized_get_representation(self, state):
    return self.get_representation(state)
  
  @nnx.jit
  @nnx.vmap(in_axes=(None, 0))
  def vectorized_decoder(self, afterstate):
    return self.decoder(afterstate)
  
  @nnx.jit
  @nnx.vmap(in_axes=(None, 0))
  def vectorized_get_policy(self, afterstate):
    return self.get_policy_outcome(afterstate)
  
  @nnx.jit
  @nnx.vmap(in_axes=(None, 0, 0))
  def vectorized_get_next_afterstate(self, afterstate, outcome):
    return self.get_next_afterstate(afterstate, outcome)

  
  def get_policy_from_real(self, state):
    """Returns a learned policy for real state. 
    Calls both the representation and policy networks.
    Used for inference."""
    afterstate = self._jit_get_representation(state)
    policy_logits, outcome = self._jit_get_policy_outcome(afterstate)
    return nnx.softmax(policy_logits)
  
  def loss_function_decode_first_step(self, state_targets, action_targets, valid):
    #Trajectory, Batch, ...] is the shape of the 
    # state and action targets
    #TODO: After checking that this is correct, try to remove the for loops
    total_loss = 0
    for i in range(state_targets.shape[0]):
      #encode the first state
      #[Batch, afterstate_dim]
      afterstates = self.vectorized_get_representation(state_targets[i])
      #[Batch, actions]
      policy_logits, outcomes = self.vectorized_get_policy(afterstates)
      decoded_states = self.vectorized_decoder(afterstates)
      total_loss += jnp.mean(optax.l2_loss(decoded_states, state_targets[i]) * valid[i])
      total_loss += jnp.mean(optax.softmax_cross_entropy_with_integer_labels(policy_logits, jnp.squeeze(action_targets[i], -1)) * valid[i])
      #VQ-VAE commitment loss
      total_loss += self.beta_commitment * jnp.mean(optax.softmax_cross_entropy(policy_logits, jax.lax.stop_gradient(outcomes)) * valid[i])
      #unrolling over the rest of the trajectory now
      for j in range(i + 1, state_targets.shape[0]):
        #only the policy losses here for now
        afterstates = self.vectorized_get_next_afterstate(afterstates, outcomes) 
        policy_logits, outcomes = self.vectorized_get_policy(afterstates)
        total_loss += jnp.mean(optax.softmax_cross_entropy_with_integer_labels(policy_logits, jnp.squeeze(action_targets[j], -1)) * valid[j])
        #VQ-VAE commitment loss
        total_loss += self.beta_commitment * jnp.mean(optax.softmax_cross_entropy(policy_logits, jax.lax.stop_gradient(outcomes)) * valid[j])
    return total_loss
  
  def loss_function_decode_all_steps(self, state_targets, action_targets, valid):
    #TODO: After checking that this is correct, try to remove the for loops
    total_loss = 0
    for i in range(state_targets.shape[0]):
      #encode the first state
      #[Batch, afterstate_dim]
      afterstates = self.vectorized_get_representation(state_targets[i])
      #[Batch, actions]
      policy_logits, outcomes = self.vectorized_get_policy(afterstates)
      decoded_states = self.vectorized_decoder(afterstates)
      total_loss += jnp.mean(optax.l2_loss(decoded_states, state_targets[i]) * valid[i])
      total_loss += jnp.mean(optax.softmax_cross_entropy_with_integer_labels(policy_logits, jnp.squeeze(action_targets[i], -1)) * valid[i])
      #VQ-VAE commitment loss
      total_loss += self.beta_commitment * jnp.mean(optax.softmax_cross_entropy(policy_logits, jax.lax.stop_gradient(outcomes)) * valid[i])
      #unrolling over the rest of the trajectory now
      for j in range(i + 1, state_targets.shape[0]):
        afterstates = self.vectorized_get_next_afterstate(afterstates, outcomes)
        decoded_states = self.vectorized_decoder(afterstates)
        total_loss += jnp.mean(optax.l2_loss(decoded_states, state_targets[j]) * valid[j])
        policy_logits, outcomes = self.vectorized_get_policy(afterstates)
        total_loss += jnp.mean(optax.softmax_cross_entropy_with_integer_labels(policy_logits, jnp.squeeze(action_targets[j], -1)) * valid[j])
        #VQ-VAE commitment loss
        total_loss += self.beta_commitment * jnp.mean(optax.softmax_cross_entropy(policy_logits, jax.lax.stop_gradient(outcomes)) * valid[j])
    return total_loss
  
  def loss_function_represent_all_steps(self, state_targets, action_targets, valid):
     #TODO: After checking that this is correct, try to remove the for loops
    total_loss = 0
    for i in range(state_targets.shape[0]):
      #encode the first state
      #[Batch, afterstate_dim]
      afterstates = self.vectorized_get_representation(state_targets[i])
      #[Batch, actions]
      policy_logits, outcomes = self.vectorized_get_policy(afterstates)
      decoded_states = self.vectorized_decoder(afterstates)
      total_loss += jnp.mean(optax.l2_loss(decoded_states, state_targets[i]) * valid[i])
      total_loss += jnp.mean(optax.softmax_cross_entropy_with_integer_labels(policy_logits, jnp.squeeze(action_targets[i], -1)) * valid[i])
      #VQ-VAE commitment loss
      total_loss += self.beta_commitment * jnp.mean(optax.softmax_cross_entropy(policy_logits, jax.lax.stop_gradient(outcomes)) * valid[i])
      #unrolling over the rest of the trajectory now
      for j in range(i + 1, state_targets.shape[0]):
        afterstates = self.vectorized_get_next_afterstate(afterstates, outcomes)
        repr_afterstates = self.vectorized_get_representation(state_targets[j])
        total_loss += jnp.mean(optax.l2_loss(afterstates, repr_afterstates) * valid[j])
        policy_logits, outcomes = self.vectorized_get_policy(afterstates)
        total_loss += jnp.mean(optax.softmax_cross_entropy_with_integer_labels(policy_logits, jnp.squeeze(action_targets[j], -1)) * valid[j])
        #VQ-VAE commitment loss
        total_loss += self.beta_commitment * jnp.mean(optax.softmax_cross_entropy(policy_logits, jax.lax.stop_gradient(outcomes)) * valid[j])
    return total_loss

  @nnx.jit   
  #This should be compiled only once!!!
  @chex.assert_max_traces(n=1)
  def optimize_step(self,state_targets, action_targets, valid):
    """Function performing the actual step, where 
    we call the loss_function, derivate it with respect to self (a.k.a the params
    stored in self), and then call self.optimizer update."""
    def loss_fn(current_train_state):
      return current_train_state.loss_function_decode_first_step(state_targets, action_targets, valid)
    
    total_loss, all_grads = nnx.value_and_grad(loss_fn)(self)
    self.optimizer.update(all_grads)
    return total_loss
    

class VQ_VAETrain:
  """Train the MuZero like algorithm
  with an additional step with the encoder, """
  def __init__(self, config:VQ_VAEConfig, num_cards: int, save_each= -1, print_each = -1, model_save_dir = ""):
    #For now only for PointCardMatching, later change
    #it for general JaxGame
    self.game = PointCardMatching(num_cards=num_cards) 
    self.config = config
    self.networks_key = jax.random.key(self.config.networks_seed)
    self.gameplay_key = jax.random.key(self.config.gameplay_seed)
    self.state_dimension = self.game.state_tensor_size()
    self.action_dimension = self.game.num_distinct_actions()
    
    
    self.optim = optax.adamw(self.config.learning_rate)

    #The object handling the actual training step is 
    # a stateful object inheriting from nnx.Module
    self.networks = VQ_VAETrainStep(config,self.action_dimension, self.state_dimension, self.optim)

    self.steps = 0

    self.get_example_timestep()
    self.prepare_checkpointer(save_each, print_each, model_save_dir)

  def prepare_checkpointer(self, save_each, print_each, model_save_dir):
    if not model_save_dir:
      model_save_dir = f"/trained_networks/point_card_matching{self.game.num_cards}/seed{self.config.gameplay_seed}/network_seed{self.config.networks_seed}"
      model_save_dir = os.getcwd() + model_save_dir
    print(F"Saving at {model_save_dir}")
    options = ocp.CheckpointManagerOptions(
          save_interval_steps=save_each if save_each > 0 else None,
          create=True
      )
    self.print_each = print_each
    self.checkpoint_manager = ocp.CheckpointManager(model_save_dir,
                                                    item_names = ("network_state", "config", "gameplay_key"),
                                                    options=options)

  def get_example_timestep(self):
      valid = jnp.zeros(1, dtype=bool)
      state = jnp.zeros((self.state_dimension))
      action = jnp.zeros(1, dtype=int)
      legal = jnp.stack([np.ones(self.action_dimension), jax.nn.one_hot(0, self.action_dimension)])
      self.example_timestep = TimeStep(valid=valid,
                                       state = state,
                                       action=action,
                                       legal=legal)

  def get_next_gameplay_key(self):
    self.gameplay_key, temp_key = jax.random.split(self.gameplay_key)
    return temp_key

  def training_step(self):
    trajectory_key = self.get_next_gameplay_key()
    trajectory_key = jax.random.split(trajectory_key, self.config.batch_size)
    sample_trajectories = jax.vmap(self.sample_trajectory, in_axes=(0), out_axes=(1))
    #[Trajectory, Batch, ...]
    batch_timestep = sample_trajectories(trajectory_key)
    #params_pre_update = nnx.variables(self.networks, nnx.Param).to_pure_dict()
    #just to ensure shape consistency with the classic batch
    step_loss = self.networks.optimize_step(batch_timestep.state, batch_timestep.action, batch_timestep.valid)
    #params_post_update = nnx.variables(self.networks, nnx.Param).to_pure_dict()
    #check_param_difference(params_post_update, params_pre_update)
    return step_loss

  def train_model(self, num_steps):
    for s in range(num_steps):
      step_loss = self.training_step()
      self.checkpoint_manager.save(self.steps, args = ocp.args.Composite(
                                                        network_state = ocp.args.StandardSave(nnx.split(self.networks)[1].to_pure_dict()),
                                                        config = ocp.args.StandardSave(self.config),
                                                        gameplay_key = ocp.args.ArraySave(self.gameplay_key)))
      self.steps += 1
      if self.print_each > 0 and self.steps % self.print_each == 0:
        print(f"Step {self.steps}, loss: {step_loss}")
      self.checkpoint_manager.wait_until_finished()
    
  def restore_latest_checkpoint(self, step: int=-1):
    """Restore the model from a given saved checkpoint.
    If step = -1 is provided, the last saved checkpoint is restored"""
    latest_step = self.checkpoint_manager.latest_step()
    assert latest_step is not None, "No network checkpoint was found!"
    assert step <= latest_step, "Given step is larger that the last saved step!"
    #We restore the saved config 
    # and network state. Then use the config to initialize self.networks anew,
    # because the provided dummy config could cause shape mismatch. Then, finally
    restore_args = ocp.args.Composite(
                        #Restores as a general PyTree
                        network_state=ocp.args.StandardRestore(None),
                        config = ocp.args.StandardRestore(self.config),
                        gameplay_key = ocp.args.ArrayRestore(self.gameplay_key))
    restore_step = step if step > -1 else latest_step
    print(f"Restoring model step {restore_step}")
    restored_items = self.checkpoint_manager.restore(restore_step, args=restore_args)
    self.config = restored_items["config"]
    #Also need to restore the last PRNG key that was used, 
    # to ensure that the trajectory sampling will not produce
    # identical data again
    self.gameplay_key = restored_items["gameplay_key"]
    #The action dimension and state dimension depend only on the game
    # so these should not be trouble. The optim potentially could be, 
    # but as long as optimizer type is not different than the saved state it should be fine
    new_networks = VQ_VAETrainStep(self.config, self.action_dimension, self.state_dimension, self.optim)
    graphdef, current_state = nnx.split(new_networks)
    saved_state = restored_items["network_state"]
    current_state.replace_by_pure_dict(saved_state)
    self.networks = nnx.merge(graphdef, current_state)
    #Remember where the training ended
    self.steps = restore_step

  @partial(jax.jit, static_argnums=0)
  def sample_trajectory(self, key) ->TimeStep:
    init_key, trajectory_key, = jax.random.split(key)
    trajectory_key = jax.random.split(trajectory_key, self.config.trajectory_max)
    

    actions = self.action_dimension
          
    game_state, legal_actions = self.game.initialize_structures(init_key)
    init_carry = SampleTrajectoryCarry(
      game_state = game_state,
      terminal = False,
      legal_actions = legal_actions
    )
    
    @jax.jit
    def choice_wrapper(key, p):
      action = jax.random.choice(key, actions, p=p)
      #action_oh = jax.nn.one_hot(action, actions)
      return action#, action_oh
    
    vectorized_sample_action = jax.vmap(choice_wrapper, in_axes=(0, 0), out_axes=0)
    
    def _sample_trajectory(carry: SampleTrajectoryCarry, xs) -> tuple[SampleTrajectoryCarry, chex.Array]:
      (key, turn) = xs
      state, p1_iset, p2_iset, public_state = self.game.get_info(carry.game_state)
      #observations not used for now
      #obs = jnp.stack((p1_iset, p2_iset), axis=0)
      #obs = jnp.where(carry.terminal, self.example_timestep.obs, obs) 
      
      state = jnp.where(carry.terminal, self.example_timestep.state, state)
      
      #Here the fixed reference policy is used
      pi = get_reference_policy(carry.game_state, carry.legal_actions)
      
      sample_key, action_key = jax.random.split(key)
      # For each player samples a single action
      sample_key = jax.random.split(sample_key, 2)
      
      action = vectorized_sample_action(sample_key, pi)
      next_game_state, next_legal, next_rewards, terminal= self.game.apply_action(carry.game_state, action_key, turn, action)
      valid = (jnp.ones_like(next_rewards, dtype=int) - carry.terminal).astype(bool)
      terminal = jnp.logical_or(terminal, carry.terminal)
      next_rewards = jnp.where(valid, next_rewards, jnp.zeros_like(next_rewards))
      new_carry = SampleTrajectoryCarry(
        game_state = next_game_state,
        terminal = terminal,
        legal_actions=jnp.where(terminal, self.example_timestep.legal, next_legal)
      )
      timestep = TimeStep(
        valid = valid,
        state = state,
        legal = carry.legal_actions,
        #Only interested in player 1 actions here
        action = action[0][None],
      )
      return new_carry, timestep
    _, timestep = jax.lax.scan(_sample_trajectory,
             init=init_carry,
             xs=(trajectory_key, jnp.arange(self.config.trajectory_max)))
    #[Trajectory, ...]
    return timestep