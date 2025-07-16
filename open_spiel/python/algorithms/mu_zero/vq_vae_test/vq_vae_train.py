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
from open_spiel.python.algorithms.mu_zero.vq_vae_test.vq_vae_networks import Afterstate_representation_function, Afterstate_dynamics_function, Afterstate_decoder_function, Policy_function, Codebook_function
from open_spiel.python.algorithms.mu_zero.vq_vae_test.train_utils import get_reference_policy, plot_param_devs




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
  codebook_hidden_size: int = 64

  learning_rate: float = 3e-4
  networks_seed: int = 99
  gameplay_seed: int = 42

@chex.dataclass(frozen=True)
class TimeStep():
  
  state_valid: chex.Array = () # [..., 1]
  action_valid: chex.Array = () # [..., 1] # These two valid are decoupled, because actions in terminal state are not defined, but state/infoset tensors should be for our loss
  state: chex.Array = () # [..., state_dim] 
  legal: chex.Array = () # [..., Player, A]
  
  action: chex.Array = () # [..., Player, A]
  #policy: chex.Array = () # [..., Player, A]
  
  #reward: chex.Array = () # [..., 1] Reward after playing an action
  
@chex.dataclass(frozen=True)
class SampleTrajectoryCarry:
  game_state: PointCardMatchingState
  terminal: bool
  valid: bool
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
    networks_key, codebook_rngs = self.get_next_network_rngs(networks_key)
    self.codebook = Codebook_function(action_dimension, config.afterstate_dimension, config.codebook_hidden_size, codebook_rngs)


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
  
  def get_next_closest_afterstate(self, afterstate, action):
    """Apply dynamics step to get the next encoded afterstate
    and then pick its closest representative from the learned codebook and return it."""
    next_code = self.get_next_afterstate(afterstate, action)
    codebook = self.get_codebook(afterstate)
    closest_index = jnp.argmin((jnp.sum((next_code[None, :] - codebook) **2, axis=-1)), axis=-1)
    #[action_dim, afterstate_dim]
    closest_index_oh = nnx.one_hot(closest_index, codebook.shape[-2], axis=-1)
    #[afterstate_dim]
    closest_representative = jnp.sum(codebook * closest_index_oh[..., None], axis=-2)
    return closest_representative
    
  
  def get_codebook(self, aftestate):
    """Returns a codebook of next possible afterstates
    of shape (action_dimension, afterstate_dimension)"""
    return self.codebook(aftestate)

  def get_policy(self, afterstate):
    policy_logits  = self.policy(afterstate)
    return policy_logits
  
  def get_next_afterstate(self, afterstate, action):
    next_afterstate = self.afterstate_dynamics(afterstate, action)
    return next_afterstate
  
  @nnx.jit
  def _jit_get_representation(self, state):
    return self.get_representation(state)
  
  @nnx.jit
  def _jit_decoder(self, afterstate):
    return self.decoder(afterstate)
  
  @nnx.jit
  def _jit_get_policy(self, afterstate):
    return self.get_policy(afterstate)
  
  @nnx.jit
  def _jit_get_next_afterstate(self, afterstate, action):
    return self.get_next_afterstate(afterstate, action)
  
  @nnx.jit
  def _jit_get_codebook(self, afterstate):
    return self.get_codebook(afterstate)
  @nnx.jit
  def _jit_get_next_closest_afterstate(self, afterstate, action):
    return self.get_next_closest_afterstate(afterstate, action)
  
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
    return self.get_policy(afterstate)
  
  @nnx.jit
  @nnx.vmap(in_axes=(None, 0, 0))
  def vectorized_get_next_afterstate(self, afterstate, action):
    return self.get_next_afterstate(afterstate, action)
  
  @nnx.jit
  @nnx.vmap(in_axes=(None, 0))
  def vectorized_get_codebook(self, afterstate):
    return self.get_codebook(afterstate)
  
  #For processing [Trajectory, Batch, ...] arrays
  @nnx.jit
  @nnx.vmap(in_axes=(None, 0))
  def trajectory_get_representation(self, state):
    return self.vectorized_get_representation(state)
  
  @nnx.jit
  @nnx.vmap(in_axes=(None, 0))
  def trajectory_decoder(self, afterstate):
    return self.vectorized_decoder(afterstate)
  
  @nnx.jit
  @nnx.vmap(in_axes=(None, 0))
  def trajectory_get_policy(self, afterstate):
    return self.vectorized_get_policy(afterstate)
  
  @nnx.jit
  @nnx.vmap(in_axes=(None, 0, 0))
  def trajectory_get_next_afterstate(self, afterstate, action):
    return self.vectorized_get_next_afterstate(afterstate, action)
  
  @nnx.jit
  @nnx.vmap(in_axes=(None, 0))
  def trajectory_get_codebook(self, afterstate):
    return self.vectorized_get_codebook(afterstate)
  
  def get_policy_from_real(self, state):
    """Returns a learned policy for real state. 
    Calls both the representation and policy networks.
    Used for inference."""
    afterstate = self._jit_get_representation(state)
    policy_logits = self._jit_get_policy(afterstate)
    return nnx.softmax(policy_logits)
  
  def loss_function_vectorized(self, timestep: TimeStep):
    l_a, l_c, l_s, l_z = 0, 0, 0, 0
    afterstates = self.trajectory_get_representation(timestep.state)
    #[Trajectory, Batch, actions]
    policy_logits = self.trajectory_get_policy(afterstates)
    #policy = nnx.softmax(policy_logits, axis=-1)
    smoothed_actions = jnp.where(timestep.action == 1, 0.95, 0.05)
    #jax.debug.breakpoint()
    #Policy loss
    l_a += jnp.mean(optax.softmax_cross_entropy(policy_logits, smoothed_actions)[..., None] * timestep.action_valid)
    #l_a += jnp.mean(optax.l2_loss(policy, smoothed_actions))
    #[Trajectory, Batch, afterstate_dim]
    # This is the output of encoder for our purposes
    next_afterstates = self.trajectory_get_next_afterstate(afterstates, timestep.action)      
    #[Trajectory, Batch, action_dim, afterstate_dim]
    codebooks = self.vectorized_get_codebook(afterstates)
    #[Trajectory, Batch]
    closest_index = jnp.argmin((jnp.sum((next_afterstates[..., None, :] - codebooks) **2, axis=-1)), axis=-1)
    #[Trajectory, Batch, action_dim]
    closest_index_oh = nnx.one_hot(closest_index, codebooks.shape[-2], axis=-1)
    #[Trajectory, Batch, afterstate_dim]
    closest_representatives = jnp.sum(codebooks * closest_index_oh[..., None], axis=-2)
    #VQ-VAE losses
    #Embedding closeness
    l_c += jnp.mean(optax.l2_loss(closest_representatives, jax.lax.stop_gradient(next_afterstates)) * timestep.action_valid)
    #Commitment loss
    l_c += self.beta_commitment * jnp.mean(optax.l2_loss(next_afterstates, jax.lax.stop_gradient(closest_representatives)) * timestep.action_valid)  
    #Applying a straight through estimator to closest_representatives to flow back to next_afterstates
    # necessary to get the gradients to flow back to the dynamics network
    closest_representatives = closest_representatives + next_afterstates - jax.lax.stop_gradient(next_afterstates) 
    #[Trajectory, Batch, state_dim]
    decoded_states = self.vectorized_decoder(closest_representatives)
    # Reconstruction loss. It is made towards the NEXT state, since
    # the dynamics are our encoder
    # The last decoded state is invalid, since it comes from an action taken at terminal
    # Similarly, the first timestep.state is not the next_state of anything
    l_s += jnp.mean(optax.l2_loss(decoded_states[:-1], timestep.state[1:]) * timestep.state_valid[1:])
    #[Trajectory - 1, Batch, afterstate_dim]
    next_representations = afterstates[1:]
    #Distance between representation and dynamics output.
    # Representation is frozen with stop_gradient for this operation
    l_z += jnp.mean(optax.l2_loss(closest_representatives[:-1], jax.lax.stop_gradient(next_representations)) * timestep.state_valid[1:])
    return l_a + l_c + l_s + l_z
  

  @nnx.jit
  def optimize_step(self,timestep: TimeStep):
    """Function performing the actual step, where 
    we call the loss_function, derivate it with respect to self (a.k.a the params
    stored in self), and then call self.optimizer update."""
    def loss_fn(current_train_state):
      return current_train_state.loss_function_vectorized(timestep)
    
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

    # self.cum_frequencies = np.zeros((4, self.action_dimension))
    self.init_std_dev = None
    self.param_devs = []
    self.iters = []

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
      action_valid = jnp.zeros(1, dtype=bool)
      state_valid = jnp.zeros(1, dtype=bool)
      state = jnp.zeros((self.state_dimension))
      action = jnp.zeros(1, dtype=int)
      legal = jnp.stack([np.ones(self.action_dimension), jax.nn.one_hot(0, self.action_dimension)])
      self.example_timestep = TimeStep(action_valid = action_valid,
                                       state_valid = state_valid,
                                       state = state,
                                       action=action,
                                       legal=legal)

  def get_next_gameplay_key(self):
    self.gameplay_key, temp_key = jax.random.split(self.gameplay_key)
    return temp_key
  
  def get_normalized_params_std_dev(self):
    """Get the standart deviation of parameters of each layer
    of each network, normalized with respect to the initial standard deviation"""
    params = nnx.variables(self.networks, nnx.Param).to_pure_dict()
    if self.init_std_dev is None:
      self.init_std_dev = jax.tree_util.tree_map(lambda x: jnp.std(x), params)
      param_devs = jax.tree_util.tree_map(lambda x: jnp.ones_like(x), self.init_std_dev)
    else: 
      param_devs = jax.tree_util.tree_map(lambda x, y: jnp.std(x) / y, params, self.init_std_dev)
    self.param_devs.append(param_devs)
    self.iters.append(self.steps)
  

  def training_step(self):
    trajectory_key = self.get_next_gameplay_key()
    trajectory_key = jax.random.split(trajectory_key, self.config.batch_size)
    sample_trajectories = jax.vmap(self.sample_trajectory, in_axes=(0), out_axes=(1))
    #[Trajectory, Batch, ...]
    batch_timestep = sample_trajectories(trajectory_key)
    #self.cum_frequencies += add_action_frequencies(batch_timestep.action)
    #params_pre_update = nnx.variables(self.networks, nnx.Param).to_pure_dict()
    step_loss = self.networks.optimize_step(batch_timestep)
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
      if self.print_each > 0 and self.steps % self.print_each == 0:
        print(f"Step {self.steps}, loss: {step_loss}")
        #self.get_normalized_params_std_dev()
      self.steps += 1
      self.checkpoint_manager.wait_until_finished()
    # print("Average action frequencies: ")
    # print(self.cum_frequencies / np.sum(self.cum_frequencies, axis=-1, keepdims=True))
    # print("Plotting param devs: ")
    # plot_param_devs(self.param_devs, self.iters)
    
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
      valid = True,
      legal_actions = legal_actions
    )
    
    @jax.jit
    def choice_wrapper(key, p):
      action = jax.random.choice(key, actions, p=p)
      action_oh = jax.nn.one_hot(action, actions)
      return action, action_oh
    
    vectorized_sample_action = jax.vmap(choice_wrapper, in_axes=(0, 0), out_axes=0)
    
    def _sample_trajectory(carry: SampleTrajectoryCarry, xs) -> tuple[SampleTrajectoryCarry, chex.Array]:
      (key, turn) = xs
      state, p1_iset, p2_iset, public_state = self.game.get_info(carry.game_state)
      #observations not used for now
      #obs = jnp.stack((p1_iset, p2_iset), axis=0)
      #obs = jnp.where(carry.terminal, self.example_timestep.obs, obs) 
      
      #Here the fixed reference policy is used
      pi = get_reference_policy(carry.game_state, carry.legal_actions)
      
      sample_key, action_key = jax.random.split(key)
      # For each player samples a single action
      sample_key = jax.random.split(sample_key, 2)
      
      action, action_oh = vectorized_sample_action(sample_key, pi)
      next_game_state, next_legal, next_rewards, terminal= self.game.apply_action(carry.game_state, action_key, turn, action)
      #Action in terminal state is not valid
      action_valid = (jnp.ones_like(next_rewards, dtype=int) - carry.terminal).astype(bool)
      #We need state tensor to be defined in terminal state as well
      state_valid = (jnp.zeros_like(next_rewards, dtype=int) + carry.valid).astype(bool) 
      next_rewards = jnp.where(action_valid, next_rewards, jnp.zeros_like(next_rewards))  
      state = jnp.where(state_valid, state, self.example_timestep.state)
      new_carry = SampleTrajectoryCarry(
        game_state = next_game_state,
        terminal = terminal,
        valid = action_valid[0],
        legal_actions=jnp.where(terminal, self.example_timestep.legal, next_legal)
      )
      timestep = TimeStep(
        action_valid = action_valid,
        state_valid = state_valid,
        state = state,
        legal = carry.legal_actions,
        #Only interested in player 1 actions here
        action = action_oh[0],
      )
      return new_carry, timestep
    _, timestep = jax.lax.scan(_sample_trajectory,
             init=init_carry,
             xs=(trajectory_key, jnp.arange(self.config.trajectory_max)))
    #[Trajectory, ...]
    return timestep