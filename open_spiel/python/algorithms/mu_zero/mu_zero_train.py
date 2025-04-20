from open_spiel.python.algorithms.rnad.rnad import EntropySchedule
from open_spiel.python.algorithms.mu_zero.flax_utils import init_network_with_optimizer, init_params_optimizer, optax_optimizer, masked_l2_loss, masked_l2_loss_with_normalization

from open_spiel.python.algorithms.rnad.rnad import RNaDSolver, RNaDConfig
from open_spiel.python.algorithms.mu_zero.jax_games.jax_game import JaxGame, GameState
from open_spiel.python.policy import TabularPolicy
from open_spiel.python.algorithms.exploitability import exploitability

from open_spiel.python.algorithms.mu_zero.muzero_networks import MAVSNetwork, SimilarityNetwork, LegalActionsNetwork, PublicStateEncoder, InfosetEncoder, PublicStateDynamicsNetwork, DynamicsNetwork, TransformationNetwork, QCriticNetwork, ExpectedNetwork, RNaDNetwork, PublicStateDecoder
from open_spiel.python.algorithms.mu_zero.train_utils import _policy_ratio,   neurd_loss, transform_trajectory_to_last_dimension, normalize_direction_with_mask,  _compute_soft_kmeans_loss_with_cluster_assignments, _compute_soft_kmeans_loss_with_single, state_v_trace, expected_v_trace, v_trace

from typing import Sequence, Any, Callable
from pyinstrument import Profiler
import jax
import jax.numpy as jnp
import jax.lax as lax

import chex
import optax

from enum import Enum

import numpy as np

import pyspiel

import functools

# Taken from RNaD original
Params = chex.ArrayTree
Optimizer = Callable[[Params, Params], Params] 

@chex.dataclass(frozen=True)
class TimeStep():
  
  valid: chex.Array = () # [..., 1]
  public_state: chex.Array = () # [..., PS]
  obs: chex.Array = () # [..., Player, O]
  legal: chex.Array = () # [..., Player, A]
  
  action: chex.Array = () # [..., Player, A]
  policy: chex.Array = () # [..., Player, A]
  
  reward: chex.Array = () # [..., 1] Reward after playing an action
 
@chex.dataclass
class Optimizers:
  rnad_optimizer: Optimizer = ()
  rnad_optimizer_target: Optimizer = ()
  expected_optimizer: Optimizer = ()
  expected_optimizer_target: Optimizer = ()
  mvs_optimizer: Optimizer = ()
  mvs_optimizer_target: Optimizer = ()
  transformation_opitimizer: Sequence[Optimizer] = () 
  abstraction_optimizer: Sequence[Optimizer]  = ()
  ps_decoder_optimizer: Sequence[Optimizer] = ()
  iset_encoder_optimizer: Sequence[Optimizer]  = ()
  similarity_optimizer: Sequence[Optimizer]  = ()
  legal_actions_optimizer: Sequence[Optimizer] = ()
  dynamics_optimizer: Optimizer = ()
  
  q_critic_optimizer: Optimizer = ()
  q_critic_optimizer_target: Optimizer = ()
 
@chex.dataclass
class NetworkParameters:
  rnad_params: Params = ()
  rnad_params_target: Params = ()
  rnad_params_prev: Params = ()
  rnad_params_prev_: Params = ()
  expected_params: Params = ()
  expected_params_target: Params = ()
  mvs_params: Params = ()
  mvs_params_target: Params = ()
  transformation_params: Sequence[Params] = () 
  abstraction_params: Sequence[Params] = ()
  ps_decoder_params: Sequence[Params] = ()
  iset_encoder_params: Sequence[Params] = ()
  similarity_params: Sequence[Params] = ()
  legal_actions_params: Sequence[Params] = ()
  dynamics_params: Params = ()
  
  q_critic_params: Params = ()
  q_critic_params_target: Params = ()
  
  
class SimilarityMetric(str, Enum):
  POLICY = "policy"
  VALUE = "value"
  POLICY_VALUE = "policy_value"
  LEGAL_ACTIONS = "legal_actions"
  LEGAL_POLICY_VALUE = "legal_policy_value"
  ACTION_HISTORY = "action_history"
  ACTION_HISTORY_POLICY = "action_history_policy"
  ACTION_HISTORY_LEGAL = "action_history_legal"
  ACTION_HISTORY_LEGAL_POLICY = "action_history_legal_policy"
  ISET_VECTOR = "iset_vector"
 
 
class DynamicsType(str, Enum):
  ISET = "iset"
  PUBLIC_STATE = "public_state"
@chex.dataclass(frozen=True)
class MuZeroTrainConfig: 
  
  batch_size: int = 32
  
  trajectory_max: int = 6
  sampling_epsilon: float = 0.0
  
  train_rnad: bool = True
  train_transformations: bool = True
  train_mvs: bool = True
  train_abstraction: bool = True
  train_dynamics: bool = True
  train_legal_actions: bool = True
  
  
  use_abstraction: bool = False
  abstraction_amount: int = 10
  abstraction_size: int = 32
  similarity_metric: SimilarityMetric = SimilarityMetric.POLICY_VALUE
  
  abstraction_soft_k_means_temperature: float = 1.0
  abstraction_soft_k_means_closeness_assignment: float = 0.5
  abstraction_soft_k_means_repulsive_force: float = 3.0
  transformation_soft_k_means_temperature: float = 1.0
  transformation_soft_k_means_closeness_assignment: float = 0.5
  transformation_soft_k_means_repulsive_force: float = 3.0
  
  dynamics_type: DynamicsType = DynamicsType.PUBLIC_STATE
  
  ps_encoder_hidden_size: int = 128
  ps_decoder_hidden_size: int = 64
  iset_hidden_size: int = 64
  dynamics_hidden_size: int = 64
  similarity_hidden_size: int = 64
  mvs_hidden_size: int = 64
  legal_actions_hidden_size: int = 64
  transformation_hidden_size: int = 128
  rnad_hidden_size: int = 256
  
  transformations: int = 10
  matrix_valued_states: bool = True
  
  c_iset_vtrace: float = 1.0
  rho_iset_vtrace: float = np.inf
  c_state_vtrace: float = 1.0
  rho_state_vtrace: float = np.inf
  
  eta_regularization: float = 0.2
  entropy_schedule_repeats: Sequence[int] = (1,)
  entropy_schedule_size: Sequence[int] = (2000,)
  
  learning_rate: float = 3e-4
  target_network_update: float = 1e-3
  seed: int = 42
  

# This contains RNaD implementation. Note that this implementation is specific for two-player zero-sum games. Unlike the open_spiel RNaD that can be used to general-sum multiplayer games.
class MuZeroTrain():
  def __init__(self, game, config: MuZeroTrainConfig) -> None:
    assert config.matrix_valued_states, "Multi-valued states are not implemented."
    self.config = config
    self.game = game
    if isinstance(self.game, JaxGame):
      print("Warning: you use Jax game, so you need to use jax_step method")
      
    self.init()
    
  def init(self):
    self.actions = self.game.num_distinct_actions()
    
    if self.config.use_abstraction:
      self.obs = self.config.abstraction_size
    else:
      self.obs = self.game.information_state_tensor_shape()
      
    # self.rng_key = jax.random.PRNGKey(self.config.seed)
    self.rng_key = jax.random.key(self.config.seed)
    
    # temp_keys = self.get_next_rng_keys(6)
    
    self.example_state  = self.new_initial_state()
    self.example_timestep = self.default_timestep()
    self.example_obs = np.ones((self.obs))
    
    self._entropy_schedule = EntropySchedule(
        sizes=self.config.entropy_schedule_size,
        repeats=self.config.entropy_schedule_repeats)
    
    self.expected_network = ExpectedNetwork(self.config.rnad_hidden_size)
    self.rnad_network = RNaDNetwork(self.config.rnad_hidden_size, self.actions)
    self.abstraction_network = PublicStateEncoder(self.config.ps_encoder_hidden_size, self.config.abstraction_size, self.config.abstraction_amount)
    self.ps_decoder = PublicStateDecoder(self.config.ps_decoder_hidden_size, self.game.public_state_tensor_shape())
    self.iset_encoder = InfosetEncoder(self.config.iset_hidden_size, self.config.abstraction_amount)
    self.similarity_network = SimilarityNetwork(self.config.similarity_hidden_size, self.similarity_output_size())
    self.legal_actions_network = LegalActionsNetwork(self.config.legal_actions_hidden_size, self.actions)
    
    if self.config.dynamics_type == DynamicsType.ISET:
      self.dynamics_network = DynamicsNetwork(self.config.dynamics_hidden_size, self.obs)
    elif self.config.dynamics_type == DynamicsType.PUBLIC_STATE:
      assert self.config.use_abstraction == True, "Dynamics for Public state work only with abstrations."
      self.dynamics_network = PublicStateDynamicsNetwork(self.config.dynamics_hidden_size, self.game.public_state_tensor_shape(), self.config.abstraction_amount)
      
    self.transformation_network = TransformationNetwork(self.config.transformation_hidden_size, self.config.transformations, self.actions)
    self.mvs_network = MAVSNetwork(self.config.mvs_hidden_size, self.config.transformations + 1)
    
    
    self._rnad_loss = jax.value_and_grad(self.rnad_loss, has_aux=False) # Deprecate this?
    self._abstraction_loss = jax.value_and_grad(self.abstraction_loss, argnums=[0,1,2,3], has_aux=False)
    self._expected_loss = jax.value_and_grad(self.expected_loss, has_aux=False)
    self._rnad_with_expected_loss = jax.value_and_grad(self.rnad_with_expected_loss, has_aux=False)
    
    if self.config.use_abstraction:
      
      if self.config.dynamics_type == DynamicsType.ISET:
        self._dynamics_loss = jax.value_and_grad(self.abstracted_dynamics_loss, has_aux=False)
      elif self.config.dynamics_type == DynamicsType.PUBLIC_STATE:
        self._dynamics_loss = jax.value_and_grad(self.abstracted_ps_dynamics_loss, has_aux=False)
        
      
      self._transformation_loss = jax.value_and_grad(self.abstracted_transformation_loss, has_aux=False)
      self._mvs_loss = jax.value_and_grad(self.abstracted_mvs_loss, has_aux=False)
      self._legal_actions_loss = jax.value_and_grad(self.abstracted_legal_actions_loss, has_aux=False)
    else:  
      self._dynamics_loss = jax.value_and_grad(self.non_abstracted_dynamics_loss, has_aux=False)
      self._transformation_loss = jax.value_and_grad(self.non_abstracted_transformation_loss, has_aux=False)
      self._mvs_loss = jax.value_and_grad(self.non_abstracted_mvs_loss, has_aux=False)
      self._legal_actions_loss = jax.value_and_grad(self.non_abstracted_legal_actions_loss, has_aux=False)
     
    
    # temp_key = self.get_next_rng_key()
    temp_keys = self.get_next_rng_keys(16)
    params = self.rnad_network.init(temp_keys[0], self.example_timestep.obs, self.example_timestep.legal)
    params_target = self.rnad_network.init(temp_keys[0], self.example_timestep.obs, self.example_timestep.legal)
    params_prev = self.rnad_network.init(temp_keys[0], self.example_timestep.obs, self.example_timestep.legal)
    params_prev_ = self.rnad_network.init(temp_keys[0], self.example_timestep.obs, self.example_timestep.legal)
    
    optimizer = optax_optimizer(params, optax.chain(optax.adam(self.config.learning_rate, b1=0.0), optax.clip(100)))
    optimizer_target = optax_optimizer(params_target, optax.sgd(self.config.target_network_update))
    
    
    
    
    # TODO: Different init?
    p1_abstraction_params = self.abstraction_network.init(temp_keys[1], self.example_timestep.public_state)
    p2_abstraction_params = self.abstraction_network.init(temp_keys[2], self.example_timestep.public_state)
    
    # TODO: Do we want 2 different networks for iset encoder and similarity?
    p1_iset_encoder_params = self.iset_encoder.init(temp_keys[3], self.example_timestep.obs)
    p2_iset_encoder_params = self.iset_encoder.init(temp_keys[4], self.example_timestep.obs)
    
    p1_ps_decoder_params = self.ps_decoder.init(temp_keys[5], np.ones((1, self.config.abstraction_size)))
    p2_ps_decoder_params = self.ps_decoder.init(temp_keys[6], np.ones((1, self.config.abstraction_size)))
    
    # Similarity always uses abstraction
    p1_similarity_params = self.similarity_network.init(temp_keys[7], np.ones((1, self.config.abstraction_size)))
    p2_similarity_params = self.similarity_network.init(temp_keys[8], np.ones((1, self.config.abstraction_size)))
    
    p1_legal_actions_params = self.legal_actions_network.init(temp_keys[9], self.example_obs)
    p2_legal_actions_params = self.legal_actions_network.init(temp_keys[10], self.example_obs)
    
    # self.dynamics_params = self.dynamics_network.init(temp_keys[6], self.example_timestep.obs, self.example_timestep.obs, self.example_timestep.action, self.example_timestep.action)
    dynamics_params = self.dynamics_network.init(temp_keys[11], self.example_obs, self.example_obs, self.example_timestep.action, self.example_timestep.action)

    mvs_params = self.mvs_network.init(temp_keys[12], self.example_obs, self.example_obs)
    mvs_params_target = self.mvs_network.init(temp_keys[12], self.example_obs, self.example_obs)
    
    p1_transformation_params = self.transformation_network.init(temp_keys[13], self.example_obs)
    p2_transformation_params = self.transformation_network.init(temp_keys[14], self.example_obs)
    
    expected_params = self.expected_network.init(temp_keys[15], self.example_timestep.obs, self.example_timestep.obs)
    expected_params_target = self.expected_network.init(temp_keys[15], self.example_timestep.obs, self.example_timestep.obs)
    
    
    p1_abstraction_optimizer = optax_optimizer(p1_abstraction_params, optax.chain(optax.adam(self.config.learning_rate), optax.clip(100)))
    p2_abstraction_optimizer = optax_optimizer(p2_abstraction_params, optax.chain(optax.adam(self.config.learning_rate), optax.clip(100)))

    p1_iset_encoder_optimizer = optax_optimizer(p1_iset_encoder_params, optax.chain(optax.adam(self.config.learning_rate), optax.clip(100)))
    p2_iset_encoder_optimizer = optax_optimizer(p2_iset_encoder_params, optax.chain(optax.adam(self.config.learning_rate), optax.clip(100)))
    
    p1_ps_decoder_optimizer = optax_optimizer(p1_ps_decoder_params, optax.chain(optax.adam(self.config.learning_rate), optax.clip(100)))
    p2_ps_decoder_optimizer = optax_optimizer(p2_ps_decoder_params, optax.chain(optax.adam(self.config.learning_rate), optax.clip(100)))
    
    p1_similarity_optimizer = optax_optimizer(p1_similarity_params, optax.chain(optax.adamw(self.config.learning_rate), optax.clip(100)))
    p2_similarity_optimizer = optax_optimizer(p2_similarity_params, optax.chain(optax.adamw(self.config.learning_rate), optax.clip(100)))
    
    p1_legal_actions_optimizer = optax_optimizer(p1_legal_actions_params, optax.chain(optax.adam(self.config.learning_rate), optax.clip(100)))
    p2_legal_actions_optimizer = optax_optimizer(p2_legal_actions_params, optax.chain(optax.adam(self.config.learning_rate), optax.clip(100)))
    
    dynamics_optimizer = optax_optimizer(dynamics_params, optax.chain(optax.adam(self.config.learning_rate), optax.clip(100)))
    
    mvs_optimizer = optax_optimizer(mvs_params, optax.chain(optax.adam(self.config.learning_rate), optax.clip(100)))
    mvs_optimizer_target = optax_optimizer(mvs_params_target, optax.sgd(self.config.target_network_update))
    
    p1_transformation_optimizer = optax_optimizer(p1_transformation_params, optax.chain(optax.adam(self.config.learning_rate), optax.clip(100)))
    p2_transformation_optimizer = optax_optimizer(p2_transformation_params, optax.chain(optax.adam(self.config.learning_rate), optax.clip(100)))
    
    
    expected_optimizer = optax_optimizer(expected_params, optax.chain(optax.adam(self.config.learning_rate), optax.clip(100)))
    expected_optimizer_target = optax_optimizer(expected_params_target, optax.sgd(self.config.target_network_update))
    
    
    
    self.optimizers = Optimizers(
      rnad_optimizer = optimizer,
      rnad_optimizer_target = optimizer_target,
      expected_optimizer = expected_optimizer,
      expected_optimizer_target = expected_optimizer_target,
      mvs_optimizer = mvs_optimizer,
      mvs_optimizer_target = mvs_optimizer_target,
      transformation_opitimizer = (p1_transformation_optimizer, p2_transformation_optimizer),
      abstraction_optimizer = (p1_abstraction_optimizer, p2_abstraction_optimizer),
      ps_decoder_optimizer= (p1_ps_decoder_optimizer, p2_ps_decoder_optimizer),
      iset_encoder_optimizer = (p1_iset_encoder_optimizer, p2_iset_encoder_optimizer),
      similarity_optimizer = (p1_similarity_optimizer, p2_similarity_optimizer),
      legal_actions_optimizer= (p1_legal_actions_optimizer, p2_legal_actions_optimizer),
      dynamics_optimizer = dynamics_optimizer
    )
    
    self.network_parameters = NetworkParameters(
      rnad_params = params,
      rnad_params_target = params_target,
      rnad_params_prev = params_prev,
      rnad_params_prev_ = params_prev_,
      expected_params = expected_params,
      expected_params_target = expected_params_target,
      mvs_params = mvs_params,
      mvs_params_target = mvs_params_target, 
      transformation_params = (p1_transformation_params, p2_transformation_params), 
      abstraction_params = (p1_abstraction_params, p2_abstraction_params),
      ps_decoder_params= (p1_ps_decoder_params, p2_ps_decoder_params),
      iset_encoder_params = (p1_iset_encoder_params, p2_iset_encoder_params),
      similarity_params = (p1_similarity_params, p1_similarity_params),
      legal_actions_params= (p1_legal_actions_params, p2_legal_actions_params),
      dynamics_params = dynamics_params
    )
    
    self.learner_steps = 0

  def similarity_output_size(self):
    if self.config.similarity_metric == SimilarityMetric.POLICY:
      return self.actions
    elif self.config.similarity_metric == SimilarityMetric.VALUE:
      return 1
    elif self.config.similarity_metric == SimilarityMetric.POLICY_VALUE:
      return self.actions + 1
    elif self.config.similarity_metric == SimilarityMetric.LEGAL_ACTIONS:
      return self.actions
    elif self.config.similarity_metric == SimilarityMetric.LEGAL_POLICY_VALUE:
      return 2 * self.actions + 1
    elif self.config.similarity_metric == SimilarityMetric.ACTION_HISTORY:
      return self.actions * self.config.trajectory_max
    elif self.config.similarity_metric == SimilarityMetric.ACTION_HISTORY_POLICY:
      return self.actions * self.config.trajectory_max + self.actions
    elif self.config.similarity_metric == SimilarityMetric.ACTION_HISTORY_LEGAL:
      return self.actions * self.config.trajectory_max + self.actions
    elif self.config.similarity_metric == SimilarityMetric.ACTION_HISTORY_LEGAL_POLICY:
      return self.actions * self.config.trajectory_max + 2 * self.actions
    elif self.config.similarity_metric == SimilarityMetric.ISET_VECTOR:
      return self.game.information_state_tensor_shape()
    assert False, "Unknown similarity metric"   

  def default_timestep(self):
    obs = np.zeros((2, self.game.information_state_tensor_shape()), dtype=np.float32)
    public_state = np.zeros(self.game.public_state_tensor_shape(), dtype=np.float32)
    
    legal = np.ones((2, self.actions), dtype=np.int8)
    action = np.ones(self.actions, dtype=np.float32)
    policy = np.ones(self.actions, dtype=np.float32)
    valid = np.array([0], dtype=np.float32)
    reward = np.array([0], dtype=np.float32)
    
    ts = TimeStep(
      valid = valid,
      public_state = public_state,
      obs = obs,
      legal = legal,
      action = action, 
      policy = policy,
      reward = reward
    )
    # return ts
    return ts

  def new_initial_state(self):
    if isinstance(self.game, JaxGame):
      start_key = self.get_next_rng_key()
      return self.game.new_initial_state(start_key)
    return self.game.new_initial_state()

  @functools.partial(jax.jit, static_argnums=(0,))
  def _jit_get_rnad_network(self, params, obs, legal) -> chex.Array:
    return self.rnad_network.apply(params, obs, legal)
  
  @functools.partial(jax.jit, static_argnums=(0,))
  def _jit_get_policy(self, params, obs, legal) -> chex.Array:
    return self._jit_get_rnad_network(params, obs, legal)[0]
  
  # TODO: Be careful, this sometimes produces an action that is illegal
  @functools.partial(jax.jit, static_argnums=(0, ))
  def _jit_sample_action(self, key, pi: chex.Array):
    
    def choice_wrapper(key, pi):
      return jax.random.choice(key, self.actions, p=pi)
    
    action = jax.vmap(choice_wrapper, in_axes=(0, 0), out_axes=0)(key, pi)
    action_oh = jax.nn.one_hot(action, self.actions)
    return action, action_oh
  
  @functools.partial(jax.jit, static_argnums=(0,))
  def _jit_get_policy_and_action(self, params, key, obs, legal) -> chex.Array:
    pi = self._jit_get_policy(params, obs, legal)
    action, action_oh = self._jit_sample_action(key, pi)
    return pi, action, action_oh
  
  @functools.partial(jax.jit, static_argnums=(0,))
  def _jit_get_batch_policy(self, params, keys, obs, legal) -> chex.Array:
    return jax.vmap(self._jit_get_policy_and_action, in_axes=(None, 1, 1, 1), out_axes=1)(params, keys, obs, legal)
  
  @functools.partial(jax.jit, static_argnums=(0,))
  def _jit_get_legal_actions(self, legal_actions_params, obs) -> chex.Array:
    return self.legal_actions_network.apply(legal_actions_params, obs)
    
  @functools.partial(jax.jit, static_argnums=(0,))
  def _jit_get_next_state(self, params, p1_iset, p2_iset, p1_action, p2_action):
    p1_action = jax.nn.one_hot(p1_action, self.actions)
    p2_action = jax.nn.one_hot(p2_action, self.actions)
    return self.dynamics_network.apply(params, p1_iset, p2_iset, p1_action, p2_action)
  
  @functools.partial(jax.jit, static_argnums=(0,))
  def _jit_get_next_state_ps(self, dynamics_params, p1_abstraction_params, p2_abstraction_params, p1_iset, p2_iset, p1_action, p2_action):
    next_ps, next_p1_dist, next_p2_dist, reward, terminal = self._jit_get_next_state(dynamics_params, p1_iset, p2_iset, p1_action, p2_action)
    next_ps = jnp.where(next_ps > 0.5, 1, 0)
    
    next_p1_isets = self._jit_get_all_abstractions(p1_abstraction_params, next_ps)
    next_p2_isets = self._jit_get_all_abstractions(p2_abstraction_params, next_ps)
    next_p1_iset = jnp.argmax(next_p1_dist, axis=-1, keepdims=True)
    next_p2_iset = jnp.argmax(next_p2_dist, axis=-1, keepdims=True)
    return jnp.squeeze(jnp.take_along_axis(next_p1_isets, next_p1_iset[..., jnp.newaxis], axis=-2), -2), jnp.squeeze(jnp.take_along_axis(next_p2_isets, next_p2_iset[..., jnp.newaxis], axis=-2), -2), reward, terminal
    
  
  @functools.partial(jax.jit, static_argnums=(0,))
  def _jit_get_all_abstractions(self, abstraction_params, public_state):
    return self.abstraction_network.apply(abstraction_params, public_state)
    
  @functools.partial(jax.jit, static_argnums=(0,))
  def _jit_get_iset_probabilities(self, iset_encoder_params, obs):
    return self.iset_encoder.apply(iset_encoder_params, obs)

  @functools.partial(jax.jit, static_argnums=(0,))
  def _jit_get_similarity(self, similarity_params, obs):
    return self.similarity_network.apply(similarity_params, obs)
  
  @functools.partial(jax.jit, static_argnums=(0,))
  def _jit_get_abstraction(self,abstraction_params,  iset_params, public_state, obs):
    abstraction = self.abstraction_network.apply(abstraction_params, public_state)
    iset = self.iset_encoder.apply(iset_params, obs)
    picked_iset = jnp.argmax(iset, axis=-1, keepdims=True)
    return jnp.squeeze(jnp.take_along_axis(abstraction, picked_iset[..., jnp.newaxis], axis=-2), -2)
  
  @functools.partial(jax.jit, static_argnums=(0,))
  def _jit_get_full_abstraction(self, abstraction_params, public_state):
    return self.abstraction_network.apply(abstraction_params, public_state)
  
  @functools.partial(jax.jit, static_argnums=(0,))
  def _jit_get_abstraction_with_iset_id(self,abstraction_params,  iset_params, public_state, obs):
    abstraction = self.abstraction_network.apply(abstraction_params, public_state)
    iset = self.iset_encoder.apply(iset_params, obs)
    picked_iset = jnp.argmax(iset, axis=-1, keepdims=True)
    return picked_iset, jnp.squeeze(jnp.take_along_axis(abstraction, picked_iset[..., jnp.newaxis], axis=-2), -2)
  
  @functools.partial(jax.jit, static_argnums=(0, ))
  def _jit_get_mvs(self, mvs_params, p1_iset, p2_iset):
    return self.mvs_network.apply(mvs_params, p1_iset, p2_iset)
  
  @functools.partial(jax.jit, static_argnums=(0, ))
  def _jit_get_decoded_public_state(self, ps_decoder_params, obs):
    return self.ps_decoder.apply(ps_decoder_params, obs)
  
  # The observaiton is already only for a given player pl
  def get_abstraction(self, public_state, obs, pl):
    if not self.config.use_abstraction:
      return obs
    return self._jit_get_abstraction(self.network_parameters.abstraction_params[pl], self.network_parameters.iset_encoder_params[pl], public_state, obs) 
  
  def get_both_abstraction(self, public_state, p1_iset, p2_iset):
    if not self.config.use_abstraction:
      return p1_iset, p2_iset
    p1_abstraction_iset = self._jit_get_abstraction(self.network_parameters.abstraction_params[0], self.network_parameters.iset_encoder_params[0], public_state, p1_iset)
    p2_abstraction_iset = self._jit_get_abstraction(self.network_parameters.abstraction_params[1], self.network_parameters.iset_encoder_params[1], public_state, p2_iset)
    return p1_abstraction_iset, p2_abstraction_iset
  
  def get_both_full_abstraction(self, public_state):
    if not self.config.use_abstraction:
      return jnp.ones((2, 1))
    p1_abstraction_distribution = self._jit_get_full_abstraction(self.network_parameters.abstraction_params[0], public_state)
    p2_abstraction_distribution = self._jit_get_full_abstraction(self.network_parameters.abstraction_params[1], public_state)
    return p1_abstraction_distribution, p2_abstraction_distribution

  def get_both_iset_probabilities(self, p1_iset, p2_iset):
    p1_abstraction_distribution = self._jit_get_iset_probabilities(self.network_parameters.iset_encoder_params[0], p1_iset)
    p2_abstraction_distribution = self._jit_get_iset_probabilities(self.network_parameters.iset_encoder_params[1], p2_iset)
    p1_abstraction_distribution = jax.nn.softmax(p1_abstraction_distribution, axis=-1)
    p2_abstraction_distribution = jax.nn.softmax(p2_abstraction_distribution, axis=-1)
    return p1_abstraction_distribution, p2_abstraction_distribution

  def get_decoded_public_state(self, obs, pl):
    return self._jit_get_decoded_public_state(self.network_parameters.ps_decoder_params[pl], obs)
  
  def get_next_state_from_abstraction(self, p1_iset, p2_iset, p1_action, p2_action):
    if self.config.dynamics_type == DynamicsType.ISET:
      return self._jit_get_next_state(self.network_parameters.dynamics_params, p1_iset, p2_iset, p1_action, p2_action)
    elif self.config.dynamics_type == DynamicsType.PUBLIC_STATE:
      return self._jit_get_next_state_ps(self.network_parameters.dynamics_params, self.network_parameters.abstraction_params[0], self.network_parameters.abstraction_params[1], p1_iset, p2_iset, p1_action, p2_action)
    assert False, "Wrong dynamics type"
  
  # Expects isets in the original game definition and action as a index of the action
  def get_next_state(self, public_state, p1_iset, p2_iset, p1_action, p2_action):
    if self.config.use_abstraction:
      p1_iset, p2_iset = self.get_both_abstraction(public_state, p1_iset, p2_iset)
    return self.get_next_state_from_abstraction(p1_iset, p2_iset, p1_action, p2_action) 
    
  def get_legal_actions(self, public_state, obs, pl):
    if self.config.use_abstraction:
      obs = self.get_abstraction(public_state, obs, pl)
    return self._jit_get_legal_actions(self.network_parameters.legal_actions_params[pl], obs)
  
  def get_both_legal_actions_from_abstraction(self, p1_iset, p2_iset):
    p1_legal = self._jit_get_legal_actions(self.network_parameters.legal_actions_params[0], p1_iset)
    p2_legal = self._jit_get_legal_actions(self.network_parameters.legal_actions_params[1], p2_iset)
    return p1_legal, p2_legal
  
  def get_both_legal_actions(self, public_state, p1_iset, p2_iset):
    if self.config.use_abstraction:
      p1_iset, p2_iset = self.get_both_abstraction(public_state, p1_iset, p2_iset)
    p1_legal = self._jit_get_legal_actions(self.network_parameters.legal_actions_params[0], p1_iset)
    p2_legal = self._jit_get_legal_actions(self.network_parameters.legal_actions_params[1], p2_iset)
    return p1_legal, p2_legal
      
  def get_mvs_from_abstraction(self, p1_iset, p2_iset):
    return self._jit_get_mvs(self.network_parameters.mvs_params_target, p1_iset, p2_iset) 
  
  def get_mvs(self, public_state, p1_iset, p2_iset):
    if self.config.use_abstraction:
      p1_iset, p2_iset = self.get_both_abstraction(public_state, p1_iset, p2_iset)
    return self._jit_get_mvs(self.network_parameters.mvs_params_target, p1_iset, p2_iset) 
 
  def get_policy(self, state: pyspiel.State, player: int):
    obs = state.information_state_tensor(player) 
    legal = state.legal_actions_mask(player)
    pi = self._jit_get_policy(self.network_parameters.rnad_params, obs, legal)
    return np.array(pi, dtype=np.float32)
   
  def get_policy_both(self, state: pyspiel.State):
    obs = [state.information_state_tensor(pl) for pl in range(2)] 
    legal = [state.legal_actions_mask(pl) for pl in range(2)]
    obs = np.array(obs, dtype=np.float32)
    legal = np.array(legal, dtype=np.int8)
    pi = self._jit_get_policy(self.network_parameters.rnad_params, obs, legal)
    pi = np.array(pi, dtype=np.float64)
    return pi[0], pi[1]
  
  def get_policy_and_value(self, state: pyspiel.State, player: int):
    obs = state.information_state_tensor(player) 
    legal = state.legal_actions_mask(player)
    pi, v, _, _ = self._jit_get_rnad_network(self.network_parameters.rnad_params, obs, legal)
    return np.array(pi, dtype=np.float64), np.array(v, dtype=np.float64)
  
  def get_policy_and_value_both(self, obs, legal):
    pi, v, _, _ = self._jit_get_rnad_network(self.network_parameters.rnad_params, obs, legal)
    return pi[0], pi[1], v[0], v[1]
  
  def get_policy_and_value_from_state_both(self, state: pyspiel.State):
    obs = [state.information_state_tensor(pl) for pl in range(2)] 
    legal = [state.legal_actions_mask(pl) for pl in range(2)]
    obs = np.array(obs, dtype=np.float32)
    legal = np.array(legal, dtype=np.int8)
    pi, v, _, _ = self._jit_get_rnad_network(self.network_parameters.rnad_params, obs, legal)
    pi = np.array(pi, dtype=np.float64)
    v = np.array(v, dtype=np.float64)
    return pi[0], pi[1], v[0], v[1]
  
  def get_both_similarities_and_probs(self, public_state: chex.Array, p1_iset: chex.Array, p2_iset:  chex.Array):
    p1_abstractions = self._jit_get_all_abstractions(self.network_parameters.abstraction_params[0], public_state)
    p2_abstractions = self._jit_get_all_abstractions(self.network_parameters.abstraction_params[1], public_state)
    p1_probs = self._jit_get_iset_probabilities(self.network_parameters.iset_encoder_params[0], p1_iset)
    p2_probs = self._jit_get_iset_probabilities(self.network_parameters.iset_encoder_params[1], p2_iset)
    p1_similarities = self._jit_get_similarity(self.network_parameters.similarity_params[0], p1_abstractions)
    p2_similarities = self._jit_get_similarity(self.network_parameters.similarity_params[1], p2_abstractions)
    return p1_abstractions, p2_abstractions, p1_probs, p2_probs, p1_similarities, p2_similarities
    
    
  # TODO: Improve this
  # Expects obs and legal to be in shape [Batch, Player, ...]
  def batch_policy_and_action(self, obs, legal):
    
    keys = self.get_next_rng_keys_dimensional(obs.shape[:2])
    keys = np.array(keys)
    pi, action, action_oh = self._jit_get_batch_policy(self.network_parameters.rnad_params, keys, obs, legal)
    # pi, action, action_oh = self._jit_get_policy_and_action(self.params, keys, obs, legal)
    pi = np.array(pi, dtype=np.float64)
    pi = pi / np.sum(pi, axis=-1, keepdims=True) # TODO: Remove this
    action = np.array(action, dtype=np.int32)
    action_oh = np.array(action_oh, dtype=np.float64)
    return pi, action, action_oh
    
  def _batch_states_as_timestep(self, states: Sequence[pyspiel.State]) -> TimeStep:
    reward = []
    p1_obs = []
    p2_obs = []
    p1_legal = []
    p2_legal = []
    valid = []
    
    for state in states:
      if state.is_terminal(): 
        p1_obs.append(self.example_state.information_state_tensor(0))
        p2_obs.append(self.example_state.information_state_tensor(1))
        p1_legal.append(self.example_state.legal_actions_mask(0))
        p2_legal.append(self.example_state.legal_actions_mask(1))
        valid.append(0)
      else: 
        p1_obs.append(state.information_state_tensor(0))
        p2_obs.append(state.information_state_tensor(1))
        p1_legal.append(state.legal_actions_mask(0))
        p2_legal.append(state.legal_actions_mask(1))
        valid.append(1)
       
    obs = np.stack((p1_obs, p2_obs), axis=1, dtype=np.float32)
    legal = np.stack((p1_legal, p2_legal), axis=1, dtype=np.int8)
    
    
    # p1_obs = np.array(p1_obs, dtype=np.float32)
    # p2_obs = np.array(p2_obs, dtype=np.float32)
    # p1_legal = np.array(p1_legal, dtype=np.int8)
    # p2_legal = np.array(p2_legal, dtype=np.int8)
    valid = np.array(valid, dtype=np.float32)  
    
    # obs = np.concatenate((p1_obs, p2_obs), axis=0)
    # legal = np.concatenate((p1_legal, p2_legal), axis=0)
    
    public_state = np.array([state.public_state_tensor() for state in states], dtype=np.float32)
    pi, action, action_oh = self.batch_policy_and_action(obs, legal)
    
    for i, state in enumerate(states):
      if state.is_terminal():
        reward.append(0)
        continue
      if action[i][0] not in state.legal_actions(0) or action[i][1] not in state.legal_actions(1):
        raise ValueError("Illegal action")
      state.apply_actions(action[i])
      reward.append(state.returns()[0])
      
    reward = np.array(reward, dtype=np.float32)
    return TimeStep(
      valid = valid,
      public_state = public_state,
      obs = obs,
      legal = legal,
      action = action_oh,
      policy = pi,
      reward = reward
    )
     
    
  # No chance in the game!
  def sample_trajectories(self) -> TimeStep:

    states = [self.game.new_initial_state() for _ in range(self.config.batch_size)]
    timesteps = []
    for _ in range(self.config.trajectory_max): 
      # list of states is passed as a reference to the list! So updates in function takes place in the original list
      timesteps.append(self._batch_states_as_timestep(states))
    
    return jax.tree_util.tree_map(lambda *xs: np.stack(xs, axis=0), *timesteps)
  
  def sample_trajectory(self, params, key) ->  TimeStep:
    init_key, trajectory_key, = jax.random.split(key)
    trajectory_key = jax.random.split(trajectory_key, self.config.trajectory_max)
    
    max_turns = self.config.trajectory_max
    actions = self.actions
    @chex.dataclass(frozen=True)
    class SampleTrajectoryCarry:
      game_state: GameState
      terminal: bool
      legal_actions: chex.Array
          
    game_state, legal_actions = self.game.initialize_structures(init_key)
    init_carry = SampleTrajectoryCarry(
      game_state = game_state,
      terminal = False,
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
      _, p1_iset, p2_iset, public_state = self.game.get_info(carry.game_state)
      obs = jnp.stack((p1_iset, p2_iset), axis=0)
      
      public_state = jnp.where(carry.terminal, self.example_timestep.public_state, public_state)
      obs = jnp.where(carry.terminal, self.example_timestep.obs, obs) 
      
      pi = self._jit_get_policy(params, obs, carry.legal_actions)
      random_pi = carry.legal_actions / jnp.sum(carry.legal_actions, axis=-1, keepdims=True)
      pi = self.config.sampling_epsilon * random_pi + (1 - self.config.sampling_epsilon) * pi 
      
      sample_key, action_key = jax.random.split(key)
      # For each player samples a single action
      sample_key = jax.random.split(sample_key, 2) 
      
      action, action_oh = vectorized_sample_action(sample_key, pi)
      
      next_game_state, terminal, next_rewards, next_legal = self.game.apply_action(carry.game_state, action_key, turn, action)
      
      valid = jnp.ones_like(next_rewards) - carry.terminal
      #TODO: This can likely be done better, couldnt get tree_where to work
      #timestep_legal = jnp.where(valid[..., None, None], carry.legal_actions, self.example_timestep.legal)
      next_rewards = jnp.where(valid, next_rewards, 0)
      new_carry = SampleTrajectoryCarry(
        game_state = next_game_state,
        terminal = terminal,
        legal_actions=jnp.where(terminal, self.example_timestep.legal, next_legal)
      )
      timestep = TimeStep(
        valid = valid,
        public_state = public_state,
        obs = obs,
        legal = carry.legal_actions,
        action = action_oh,
        policy = pi,
        reward = next_rewards
      )
      return new_carry, timestep
    _, timestep = lax.scan(_sample_trajectory,
             init=init_carry,
             xs=(trajectory_key, jnp.arange(max_turns)))
    return timestep
     
  
  def get_next_rng_key(self):
    self.rng_key, key = jax.random.split(self.rng_key)
    return key

  def get_next_rng_keys(self, n):
    self.rng_key, *keys = jax.random.split(self.rng_key, n+1)
    return keys
  
  # First it generates keys for the batch
  def get_next_rng_keys_dimensional(self, n):
    key = self.get_next_rng_key()
    keys = jax.random.split(key, n)
    return keys
    
  
  def rnad_loss(
    self,
    params: Params,
    params_target: Params,
    params_prev: Params,
    params_prev_: Params,
    timestep: TimeStep,
    alpha: float,
  ):
    
    # We map over trajectory dimension and player dimension
    vectorized_net_apply = jax.vmap(jax.vmap(self.rnad_network.apply, in_axes=(None, 0, 0), out_axes=0), in_axes=(None, -2, -2), out_axes=-2)
    
    pi, v, log_pi, logit = vectorized_net_apply(params, timestep.obs, timestep.legal)
    
    _, v_target, _, _ = vectorized_net_apply(params_target, timestep.obs, timestep.legal)
    _, _, log_pi_prev, _ = vectorized_net_apply(params_prev, timestep.obs, timestep.legal)
    _, _, log_pi_prev_, _ = vectorized_net_apply(params_prev_, timestep.obs, timestep.legal)
    
    # This creates the regularization term for rewards
    regularized_term = log_pi - (alpha * log_pi_prev + (1 - alpha) * log_pi_prev_) 
    
    expanded_valid = jnp.expand_dims(timestep.valid, (-2, -1))
    
    v_train_target, q_value = v_trace(v_target, expanded_valid, timestep.policy, pi, regularized_term, timestep.action, timestep.reward, c=self.config.c_iset_vtrace, rho=self.config.rho_iset_vtrace, eta=self.config.eta_regularization)
    
    # We multiply by 2, since each player acts
    normalization = jnp.sum(timestep.valid) * 2 
    v_loss = jnp.sum((expanded_valid * (v - lax.stop_gradient(v_train_target)) ** 2)) / (normalization + (normalization == 0))
    
    # Each Q is multiplied by product of importance_sampling of opponent and inverted sampling policy by the acting player.
    # This computes counterfactual reach probabilities
    sampling_policy = jnp.sum(timestep.policy * timestep.action, axis=-1, keepdims=True)
    network_policy = jnp.sum(pi * timestep.action, axis=-1, keepdims=True)
    
    
    # We do not take into account the player reaches, since infoset is always reached with the same prob
    sampling_policy = jnp.prod(sampling_policy, axis=-2, keepdims=True)
    
    # # TODO: what about invalid turns?
    importance_sampling = network_policy / sampling_policy
    
    importance_sampling = jnp.concatenate((jnp.ones((1, *importance_sampling.shape[1:])), importance_sampling[:-1]), axis=0)
    importance_sampling = jnp.cumprod(importance_sampling, axis=0)
    importance_sampling = jnp.flip(importance_sampling, axis=-2)
    
    
    loss_neurd = neurd_loss(logit, pi, q_value, timestep.legal, importance_sampling)
    
    neurd_loss_value = -jnp.sum(loss_neurd * expanded_valid) / (normalization + (normalization == 0))
    return v_loss + neurd_loss_value
   
    
    
    
    
   
  def non_abstracted_transformation_loss(self, 
                                         transformation_params: Params,
                                         abstraction_params: Params,
                                         iset_encoder_params: Params,
                                         pi_before: chex.Array,
                                         pi_after: chex.Array,
                                         public_state: chex.Array,
                                         obs: chex.Array,
                                         legal: chex.Array,
                                         valid: chex.Array):
    return self.transformation_loss(transformation_params, pi_before, pi_after, obs, legal, valid)
   
  def abstracted_transformation_loss(self,
                                     transformation_params: Params,
                                     abstraction_params: Params,
                                     iset_encoder_params: Params,
                                     pi_before: chex.Array,
                                     pi_after: chex.Array,
                                     public_state: chex.Array,
                                     obs: chex.Array,
                                     legal: chex.Array,
                                     valid: chex.Array):
    
    vectorized_abstraction = jax.vmap(self._jit_get_abstraction, in_axes=(None, None, 0, 0), out_axes=0)
    current_iset = vectorized_abstraction(abstraction_params, iset_encoder_params, public_state, obs)
    
    return self.transformation_loss(transformation_params, pi_before, pi_after, current_iset, legal, valid)
  
  def transformation_loss(self,
                          transformation_params: Params,
                          pi_before: chex.Array,
                          pi_after: chex.Array,
                          obs: chex.Array,
                          legal: chex.Array,
                          valid: chex.Array):
      
    vectorized_transformation = jax.vmap(self.transformation_network.apply, in_axes=(None, 0), out_axes=0)
    
    predicted_direction = vectorized_transformation(transformation_params, obs)
    
    update_direction = (pi_after - pi_before)
    
    mask = legal * valid[..., jnp.newaxis]
    
    predicted_direction = normalize_direction_with_mask(predicted_direction, mask[..., jnp.newaxis, :])
    update_direction = normalize_direction_with_mask(update_direction, mask)

    # TODO: This makes the whole trajectory into a single policy vector. Shall we do it this way? Maybe compare it with the old implementation
    predicted_direction = transform_trajectory_to_last_dimension(predicted_direction)
    update_direction = transform_trajectory_to_last_dimension(update_direction)
    valid_clusters = jnp.ones(1)
    
    loss, _ = _compute_soft_kmeans_loss_with_cluster_assignments(update_direction,
                                                                 predicted_direction,
                                                                 valid_clusters,
                                                                 self.config.transformation_soft_k_means_temperature,
                                                                 self.config.transformation_soft_k_means_closeness_assignment,
                                                                 self.config.transformation_soft_k_means_repulsive_force)
    return loss
  
  
  def non_abstracted_mvs_loss(self, 
                              mvs_params: Params,
                              mvs_params_target: Params,
                              policy_params: Params,
                              transformation_params: tuple[Params, Params],
                              abstraction_params: tuple[Params, Params],
                              iset_encoder_params: tuple[Params, Params],
                              timestep: TimeStep):
    
    return self.mvs_loss(mvs_params, mvs_params_target, policy_params, transformation_params, timestep.obs[..., 0, :], timestep.obs[..., 1, :], timestep)
    
  def abstracted_mvs_loss(self,
                          mvs_params: Params,
                          mvs_params_target: Params,
                          policy_params: Params,
                          transformation_params: tuple[Params, Params],
                          abstraction_params: tuple[Params, Params],
                          iset_encoder_params: tuple[Params, Params],
                          timestep: TimeStep):
    
    
    vectorized_abstraction = jax.vmap(self._jit_get_abstraction, in_axes=(None, None, 0, 0), out_axes=0)
    
    p1_current_iset = vectorized_abstraction(abstraction_params[0], iset_encoder_params[0], timestep.public_state, timestep.obs[..., 0, :])
    p2_current_iset = vectorized_abstraction(abstraction_params[1], iset_encoder_params[1], timestep.public_state, timestep.obs[..., 1, :])

    
    return self.mvs_loss(mvs_params, mvs_params_target, policy_params, transformation_params, p1_current_iset, p2_current_iset, timestep)
  
  # TODO: This is only matrix-valued states now
  def mvs_loss(self,
               mvs_params: Params,
               mvs_params_target: Params,
               rnad_params: Params,
               transformation_params: tuple[Params, Params],
               p1_obs: chex.Array,
               p2_obs: chex.Array,
               timestep: TimeStep):
    
    vectorized_policy = jax.vmap(jax.vmap(self.rnad_network.apply, in_axes=(None, 0, 0), out_axes=0), in_axes=(None, -2, -2), out_axes=-2)
    vectorized_transformation = jax.vmap(self.transformation_network.apply, in_axes=(None, 0), out_axes=0)
    vectorized_mvs = jax.vmap(self.mvs_network.apply, in_axes=(None, 0, 0), out_axes=0)
    pi, _, _, _ = vectorized_policy(rnad_params, timestep.obs, timestep.legal)
    
    
    mvs = vectorized_mvs(mvs_params, p1_obs, p2_obs)
    mvs_target = vectorized_mvs(mvs_params_target, p1_obs, p2_obs)
    
    p1_transformation_direction = vectorized_transformation(transformation_params[0], p1_obs)
    p2_transformation_direction = vectorized_transformation(transformation_params[1], p2_obs)
    

    # Dimension [Trajectory, Batch, Transformation, Player, Ations]
    transformation_direction = jnp.stack((p1_transformation_direction, p2_transformation_direction), axis=-2)
    
    
    transformation_direction = normalize_direction_with_mask(transformation_direction, jnp.expand_dims(timestep.legal * timestep.valid[..., jnp.newaxis, jnp.newaxis], -3))
    
    transformation_direction = jnp.concatenate((jnp.expand_dims(jnp.zeros_like(pi), -3), transformation_direction), -3)
    
    policy_transformations = jnp.expand_dims(pi, -3) + transformation_direction    
    policy_transformations = jnp.maximum(policy_transformations, 1e-12) # To invalidate negative actions and zeros.
    
    # Invalid actions ?
    policy_transformations = policy_transformations / jnp.sum(policy_transformations, axis=-1, keepdims=True)
     
    mvs_train_target = state_v_trace(mvs_target, timestep.policy, policy_transformations, timestep.action, timestep.valid, timestep.reward, c=self.config.c_state_vtrace, rho=self.config.rho_state_vtrace)
    
    # mask = timestep.valid[..., jnp.newaxis, jnp.newaxis]

    loss_v = timestep.valid[..., jnp.newaxis, jnp.newaxis] * (mvs - lax.stop_gradient(mvs_train_target)) ** 2
    normalization = jnp.sum(timestep.valid) * ((self.config.transformations + 1)  ** 2)
    loss_v = jnp.sum(loss_v) / (normalization + (normalization == 0))
    
    
    return loss_v
  
  def abstraction_loss(self,
                       abstraction_params: Params,
                       ps_decoder_params: Params,
                       iset_encoder_params: Params, 
                       similarity_params: Params, 
                       similarity_target: chex.Array,
                       public_state: chex.Array,
                       obs: chex.Array,
                       valid: chex.Array): 
    
    vectorized_abstraction = jax.vmap(self.abstraction_network.apply, in_axes=(None, 0), out_axes=0)
    vectorized_ps_decoder = jax.vmap(jax.vmap(self.ps_decoder.apply, in_axes=(None, 0), out_axes=0), in_axes=(None, -2), out_axes=-2)
    vectorized_iset_encoder = jax.vmap(self.iset_encoder.apply, in_axes=(None, 0), out_axes=0) 
    vectorized_similarity = jax.vmap(jax.vmap(self.similarity_network.apply, in_axes=(None, 0), out_axes=0), in_axes=(None, -2), out_axes=-2)
    
    

    abstraction = vectorized_abstraction(abstraction_params, public_state)
    decoded_ps = vectorized_ps_decoder(ps_decoder_params, abstraction)
    # Do we need this here?
    iset_probs = vectorized_iset_encoder(iset_encoder_params, obs)
    similarity = vectorized_similarity(similarity_params, abstraction)  
    
    ps_loss = (jnp.expand_dims(public_state, -2) - decoded_ps) * valid[...,None, None]
    ps_loss = jnp.mean(ps_loss ** 2)
    
    # This computes the kmeans loss and the iset loss. TODO: Add the weighted term to the pi/v distance 
    return _compute_soft_kmeans_loss_with_single(similarity_target,
                                                 similarity,
                                                 iset_probs,
                                                 valid,
                                                 self.config.abstraction_soft_k_means_temperature, 
                                                 self.config.abstraction_soft_k_means_closeness_assignment,
                                                 self.config.abstraction_soft_k_means_repulsive_force
                                                 ) + ps_loss
    
  def non_abstracted_legal_actions_loss(self,
                                        legal_actions_params: Params,
                                        abstraction_params: Params,
                                        iset_encoder_params: Params,
                                        public_state: chex.Array,
                                        obs: chex.Array,
                                        legal: chex.Array,
                                        valid: chex.Array
                                        ):
    return self.legal_actions_loss(legal_actions_params, obs, legal, valid)
    pass

  def abstracted_legal_actions_loss(self,
                                    legal_actions_params: Params,
                                    abstraction_params: Params,
                                    iset_encoder_params: Params,
                                    public_state: chex.Array,
                                    obs: chex.Array,
                                    legal: chex.Array,
                                    valid: chex.Array
                                    ):
    
    vectorized_abstraction = jax.vmap(self._jit_get_abstraction, in_axes=(None, None, 0, 0), out_axes=0)
    abstracted_obs = vectorized_abstraction(abstraction_params, iset_encoder_params, public_state, obs) 
    return self.legal_actions_loss(legal_actions_params, abstracted_obs, legal, valid)
  
  def legal_actions_loss(
    self,
    legal_actions_params: Params, 
    obs: chex.Array,
    legal: chex.Array,
    valid: chex.Array
  ):
    vectorized_legal_actions = jax.vmap(self.legal_actions_network.apply, in_axes=(None, 0), out_axes=0)
    legal_actions = vectorized_legal_actions(legal_actions_params, obs)
    
    loss = optax.losses.sigmoid_binary_cross_entropy(legal_actions, legal) * valid[..., None]
    loss = jnp.sum(loss) / jnp.sum(valid)
    # loss = jnp.sum((legal - legal_actions) ** 2) / jnp.sum(valid)
    return loss
    
  # The abstraction and iset params are just for consistency
  def non_abstracted_dynamics_loss(self, 
                                   dynamics_params: Params,
                                   abstraction_params: tuple[Params, Params],
                                   iset_encoder_params: tuple[Params, Params],
                                   timestep: TimeStep):
    
    
    
    return self.dynamics_loss(dynamics_params, timestep.obs, timestep.action, timestep.valid, timestep.reward)

    
  def abstracted_dynamics_loss(self,
                               dynamics_params: Params, 
                               abstraction_params: tuple[Params, Params], 
                               iset_encoder_params: tuple[Params, Params], 
                               timestep: TimeStep):
    
    vectorized_abstraction = jax.vmap(self._jit_get_abstraction, in_axes=(None, None, 0, 0), out_axes=0)
    
    p1_current_iset = vectorized_abstraction(abstraction_params[0], iset_encoder_params[0], timestep.public_state, timestep.obs[..., 0, :])
    p2_current_iset = vectorized_abstraction(abstraction_params[1], iset_encoder_params[1], timestep.public_state, timestep.obs[..., 1, :])
    
    return self.dynamics_loss(dynamics_params, jnp.stack([p1_current_iset, p2_current_iset], -2), timestep.action, timestep.valid, timestep.reward)

  def dynamics_loss(self,
                    dynamics_params: Params,
                    obs: chex.Array,
                    action: chex.Array, 
                    valid: chex.Array, 
                    reward: chex.Array):
    
    
    # Dynamics outputs reward for both players.
    reward = jnp.stack((reward, -reward), axis=-1)

    vectorized_dynamics = jax.vmap(self.dynamics_network.apply, in_axes=(None, 0, 0, 0, 0), out_axes=(0, 0, 0, 0))
    @chex.dataclass(frozen=True)
    class DynamicsCarry:  
      p1_iset: chex.Array
      p2_iset: chex.Array
      
    
    def _dynamics_step(carry: DynamicsCarry, xs): 
      next_p1_iset, next_p2_iset, next_reward, is_terminal = vectorized_dynamics(dynamics_params, carry.p1_iset, carry.p2_iset, action[..., 0, :], action[..., 1, :]) 
      
      
      
      new_carry = DynamicsCarry(
        p1_iset = jnp.roll(next_p1_iset, shift=1, axis=0),
        p2_iset = jnp.roll(next_p2_iset, shift=1, axis=0)
      )
      return new_carry, (jnp.roll(carry.p1_iset, shift=-1, axis=0),
                         jnp.roll(carry.p2_iset, shift=-1, axis=0),
                         next_p1_iset, next_p2_iset, next_reward, is_terminal)
    
    init_carry = DynamicsCarry( 
      p1_iset = obs[..., 0, :],
      p2_iset = obs[..., 1, :] 
    )
    
    # The result shape is [T, T, B, ...], the first T corresponds to the passes through the dynamics network. Second T corresponds to the trajectory. As an example [A, C] is if the state in trajectory (id C-A) was passed A-times through the network, so it predicts C-th state in the same trajectory.
    _, (target_p1_iset, target_p2_iset, predicted_p1_iset, predicted_p2_iset, predicted_rewards, predicted_terminals) = lax.scan(f=_dynamics_step, init=init_carry, xs = None, length=self.config.trajectory_max)
    
    # Just shifts the valid by one to the left
    valid_prediction = lax.pad(valid, 0.0, [(0, 1, 0), (0, 0, 0)])[1:]
    
    # The original - shifted finds where they change. That is the last non-terminal state
    terminal = valid - valid_prediction
    
  
    valid_dynamics = jnp.tri(self.config.trajectory_max).T
    
    
    valid_prediction = valid_dynamics[..., None] * valid_prediction[None, ...]
    valid_dynamics = valid_dynamics[..., None] * valid[None, ...]
    
    normalization = jnp.sum(valid_dynamics)

    prediction_normalization = jnp.sum(valid_prediction)
    
    p1_iset_loss = masked_l2_loss_with_normalization(predicted_p1_iset, target_p1_iset, valid_prediction[..., None], prediction_normalization)
    p2_iset_loss = masked_l2_loss_with_normalization(predicted_p2_iset, target_p2_iset, valid_prediction[..., None], prediction_normalization)
  
    reward_loss = masked_l2_loss_with_normalization(predicted_rewards, reward[None, ...], valid_dynamics[..., None], normalization) 
    terminal_loss = optax.sigmoid_binary_cross_entropy(jnp.squeeze(predicted_terminals), terminal[None, ...]) * valid_dynamics
    terminal_loss = jnp.sum(terminal_loss) / (normalization + (normalization == 0))
    
    
    return p1_iset_loss + p2_iset_loss + 5 * (reward_loss + terminal_loss)
    
  
  def dynamics_loss_single_step(self, 
                    dynamics_params: Params,
                    obs: chex.Array,
                    action: chex.Array, 
                    valid: chex.Array, 
                    reward: chex.Array):
    
    
    
    non_terminal = lax.pad(valid, 0.0, [(0, 1, 0), (0, 0, 0)])[1:]
    reward = jnp.stack((reward, -reward), axis=-1)
    
    vectorized_dynamics = jax.vmap(self.dynamics_network.apply, in_axes=(None, 0, 0, 0, 0), out_axes=(0, 0, 0, 0))
    
    next_p1_iset, next_p2_iset, next_reward, is_terminal = vectorized_dynamics(dynamics_params, obs[..., 0, :], obs[..., 1, :], action[..., 0, :], action[..., 1, :]) 
    
    next_state = jnp.stack((next_p1_iset, next_p2_iset), axis=-2)
    
    real_next_state = jnp.roll(obs, shift=-1, axis=0)
    
    
    dynamics_normalization = jnp.sum(non_terminal)
    normalization = jnp.sum(valid)
    
    dynamics_loss = (lax.stop_gradient(real_next_state) - next_state) * non_terminal[..., None, None]
    dynamics_loss = jnp.sum(dynamics_loss ** 2) / (dynamics_normalization + (dynamics_normalization == 0))
    
    reward_loss = ((lax.stop_gradient(reward) - next_reward) ** 2) * valid[..., None]
    reward_loss =  jnp.sum(reward_loss) / (normalization + (normalization == 0))
    
    terminal_loss = optax.sigmoid_binary_cross_entropy(jnp.squeeze(is_terminal),  lax.stop_gradient(1 - non_terminal)) * valid
    terminal_loss = jnp.sum(terminal_loss) / (normalization + (normalization == 0))
    
    # return reward_loss + terminal_loss
    return dynamics_loss + 7 * reward_loss + 7 * terminal_loss
      
  def abstracted_ps_dynamics_loss(self,
                               dynamics_params: Params, 
                               abstraction_params: tuple[Params, Params], 
                               iset_encoder_params: tuple[Params, Params], 
                               timestep: TimeStep):
    
    vectorized_abstraction = jax.vmap(self._jit_get_abstraction_with_iset_id, in_axes=(None, None, 0, 0), out_axes=(0, 0)) 
    
    p1_iset_id, p1_abstracted_iset = vectorized_abstraction(abstraction_params[0], iset_encoder_params[0], timestep.public_state, timestep.obs[..., 0, :])
    p2_iset_id, p2_abstracted_iset = vectorized_abstraction(abstraction_params[1], iset_encoder_params[1], timestep.public_state, timestep.obs[..., 1, :])
     
    non_terminal = lax.pad(timestep.valid, 0.0, [(0, 1, 0), (0, 0, 0)])[1:]
    reward = jnp.stack((timestep.reward, -timestep.reward), axis=-1)
    
    vectorized_dynamics = jax.vmap(self.dynamics_network.apply, in_axes=(None, 0, 0, 0, 0), out_axes=(0, 0, 0, 0, 0))
    
    next_ps, next_p1_dist, next_p2_dist, next_reward, is_terminal = vectorized_dynamics(dynamics_params, p1_abstracted_iset, p2_abstracted_iset, timestep.action[..., 0, :], timestep.action[..., 1, :]) 

    real_p1_iset_id = jnp.squeeze(jnp.roll(p1_iset_id, shift=-1, axis=0), axis=-1)
    real_p2_iset_id = jnp.squeeze(jnp.roll(p2_iset_id, shift=-1, axis=0), axis=-1) 
    
    
    real_ps = jnp.roll(timestep.public_state, shift=-1, axis=0)
    
    dynamics_normalization = jnp.sum(non_terminal)
    normalization = jnp.sum(timestep.valid)
    
    ps_loss = (lax.stop_gradient(real_ps) - next_ps) * non_terminal[..., None]
    
    ps_loss = jnp.sum(jnp.mean(ps_loss ** 2, -1)) / (dynamics_normalization + (dynamics_normalization == 0))
    
    p1_iset_loss = optax.softmax_cross_entropy_with_integer_labels(next_p1_dist, lax.stop_gradient(real_p1_iset_id)) * non_terminal
    p2_iset_loss = optax.softmax_cross_entropy_with_integer_labels(next_p2_dist, lax.stop_gradient(real_p2_iset_id)) * non_terminal
    
    p1_iset_loss = jnp.sum(p1_iset_loss) / (dynamics_normalization + (dynamics_normalization == 0))
    p2_iset_loss = jnp.sum(p2_iset_loss) / (dynamics_normalization + (dynamics_normalization == 0))
    
    reward_loss = ((lax.stop_gradient(reward) - next_reward) ** 2) * timestep.valid[..., None]
    reward_loss =  jnp.sum(reward_loss) / (normalization + (normalization == 0))
    
    terminal_loss = optax.sigmoid_binary_cross_entropy(jnp.squeeze(is_terminal),  lax.stop_gradient(1 - non_terminal)) * timestep.valid
    terminal_loss = jnp.sum(terminal_loss) / (normalization + (normalization == 0))
    
    # return reward_loss + terminal_loss
    return ps_loss + p1_iset_loss + p2_iset_loss + reward_loss +  terminal_loss
      
      
  def update_rnad(
    self,
    rnad_params: Params,
    rnad_params_target: Params,
    rnad_params_prev: Params,
    rnad_params_prev_: Params,
    optimizers: Optimizers,
    timestep: TimeStep,
    alpha: float,
    update_net: bool 
  ):
    loss, grad = self._rnad_loss(rnad_params, rnad_params_target, rnad_params_prev, rnad_params_prev_, timestep, alpha)
    
    rnad_params = optimizers.rnad_optimizer(rnad_params, grad)
    
    rnad_params_target = optimizers.rnad_optimizer_target(
        rnad_params_target, jax.tree.map(lambda a, b: a - b, rnad_params_target, rnad_params))
    
    rnad_params_prev, rnad_params_prev_ = jax.lax.cond(
        update_net,
        lambda: (rnad_params_target, rnad_params_prev),
        lambda: (rnad_params_prev, rnad_params_prev_))
    return rnad_params, rnad_params_target, rnad_params_prev, rnad_params_prev_, optimizers
      
  def update_abstraction(
    self,
    abstraction_params: tuple[Params, Params],
    ps_decoder_params: tuple[Params, Params],
    iset_encoder_params: tuple[Params, Params],
    similarity_params: tuple[Params, Params],
    optimizers: Optimizers,
    similarity: chex.Array, # Expects to have 2nd to last dimension for playe
    timestep: TimeStep,
  ):
    if not self.config.train_abstraction:
      return abstraction_params, ps_decoder_params, iset_encoder_params, similarity_params, optimizers, [0.0, 0.0]
    abs_grad = []
    abs_losses = []
    for pl in range(2):
      abstraction_loss, abstraction_grad = self._abstraction_loss(
        abstraction_params[pl], 
        ps_decoder_params[pl],
        iset_encoder_params[pl], 
        similarity_params[pl], 
        jax.lax.stop_gradient(similarity[..., pl, :]), 
        timestep.public_state, 
        timestep.obs[..., pl, :],
        timestep.valid)
       
      
      abs_losses.append(abstraction_loss)
      abs_grad.append(abstraction_grad)
    
    abstraction_params = (*[optimizers.abstraction_optimizer[pl](abstraction_params[pl], abs_grad[pl][0]) for pl in range(2)],)
    ps_decoder_params = (*[optimizers.ps_decoder_optimizer[pl](ps_decoder_params[pl], abs_grad[pl][1]) for pl in range(2)],)
    iset_encoder_params = (*[optimizers.iset_encoder_optimizer[pl](iset_encoder_params[pl], abs_grad[pl][2]) for pl in range(2)],)
    similarity_params = (*[optimizers.similarity_optimizer[pl](similarity_params[pl], abs_grad[pl][3]) for pl in range(2)],)
    
    return abstraction_params, ps_decoder_params, iset_encoder_params, similarity_params, optimizers, abs_losses
  
  def update_mvs(
    self,
    mvs_params: Params,
    mvs_params_target: Params,
    transformation_params: tuple[Params, Params],
    policy_params: Params,
    abstraction_params: tuple[Params, Params],
    iset_encoder_params: tuple[Params, Params],
    optimizers: Optimizers,
    timestep: TimeStep,
  ):
    if not self.config.train_mvs:
      return mvs_params, mvs_params_target, 0.0
    mvs_loss, mvs_grad = self._mvs_loss(
      mvs_params,
      mvs_params_target,
      policy_params,
      transformation_params,
      abstraction_params,
      iset_encoder_params,
      timestep
    )
    mvs_params = optimizers.mvs_optimizer(mvs_params, mvs_grad)
    mvs_params_target = optimizers.mvs_optimizer_target(mvs_params_target, jax.tree.map(lambda a, b: a - b, mvs_params_target, mvs_params))
    
    return mvs_params, mvs_params_target, optimizers, mvs_loss
     
  def update_transformations(
    self,
    transformation_params: tuple[Params, Params],
    abstraction_params: tuple[Params, Params],
    iset_encoder_params: tuple[Params, Params],
    optimizers: Optimizers,
    pi_before_train: chex.Array,
    pi_after_train: chex.Array,
    timestep: TimeStep,
    
  ):
    if not self.config.train_transformations:
      return transformation_params, optimizers, [0.0, 0.0]
    transform_grad = []
    losses = []
    for pl in range(2): 
      transformation_loss, transformation_grad = self._transformation_loss(
        transformation_params[pl],
        abstraction_params[pl], 
        iset_encoder_params[pl], 
        pi_before_train[..., pl, :], 
        pi_after_train[..., pl, :],
        timestep.public_state,
        timestep.obs[..., pl, :],
        timestep.legal[..., pl, :],
        timestep.valid
      )
      losses.append(transformation_loss)
      transform_grad.append(transformation_grad)
     
    transformation_params = (*[optimizers.transformation_opitimizer[pl](transformation_params[pl], transform_grad[pl]) for pl in range(2)],)
    
    return transformation_params, optimizers, losses
  
  def update_legal_actions(
    self,
    legal_actions_params: tuple[Params, Params],
    abstraction_params: tuple[Params, Params],
    iset_encoder_params: tuple[Params, Params],
    optimizers: Optimizers,
    timestep: TimeStep
  ):
    if not self.config.train_legal_actions:
      return legal_actions_params, optimizers, [0.0, 0.0]
    legal_actions_grads = []
    legal_actions_losses = []
    for pl in range(2):
      legal_actions_loss, legal_actions_grad = self._legal_actions_loss(
        legal_actions_params[pl],
        abstraction_params[pl],
        iset_encoder_params[pl],
        timestep.public_state,
        timestep.obs[..., pl, :],
        timestep.legal[..., pl, :],
        timestep.valid
      )
      legal_actions_grads.append(legal_actions_grad)
      legal_actions_losses.append(legal_actions_loss)
      
    legal_actions_params = (*[optimizers.legal_actions_optimizer[pl](legal_actions_params[pl], legal_actions_grads[pl]) for pl in range(2)],)
    return legal_actions_params, optimizers, legal_actions_losses
      
    
  
  def update_dynamics(
    self,
    dynamics_params: Params,
    abstraction_params: tuple[Params, Params],
    iset_encoder_params: tuple[Params, Params],
    optimizers: Optimizers,
    timestep: TimeStep
  ):
    if not self.config.train_dynamics:
      return dynamics_params, optimizers, 0.0
    
    dynamics_loss, dynamics_grad = self._dynamics_loss(dynamics_params, abstraction_params, iset_encoder_params, timestep)
    
    dynamics_params = optimizers.dynamics_optimizer(dynamics_params, dynamics_grad)
    
    return dynamics_params, optimizers, dynamics_loss
  
  
  @functools.partial(jax.jit, static_argnums=(0,))
  def update_parameters(
    self,
    network_parameters: NetworkParameters,
    optimizers: Optimizers,
    timestep: TimeStep,
    alpha: float,
    update_net: bool
  ):
    
    expected_params, expected_params_target, optimizers, expected_loss = self.update_expected(
      network_parameters.expected_params,
      network_parameters.expected_params_target,
      network_parameters.rnad_params,
      network_parameters.rnad_params_prev,
      network_parameters.rnad_params_prev_,
      optimizers,
      timestep,
      alpha
    )
    
      
    
    vectorized_net_apply = jax.vmap(jax.vmap(self.rnad_network.apply, in_axes=(None, 0, 0), out_axes=0), in_axes=(None, -2, -2), out_axes=-2)
    
    pi_before_train, _, _, _ = vectorized_net_apply(network_parameters.rnad_params, timestep.obs, timestep.legal) 
    
    
    rnad_params, rnad_params_target, rnad_params_prev, rnad_params_prev_, optimizers, rnad_loss = self.update_rnad_with_expected(
      network_parameters.rnad_params,
      network_parameters.rnad_params_target,
      network_parameters.rnad_params_prev,
      network_parameters.rnad_params_prev_,
      expected_params,
      expected_params_target,
      optimizers,
      timestep,
      alpha,
      update_net
    )
    # rnad_params, rnad_params_target, rnad_params_prev, rnad_params_prev_, optimizers = self.update_rnad(
    #   network_parameters.rnad_params, 
    #   network_parameters.rnad_params_target,
    #   network_parameters.rnad_params_prev,
    #   network_parameters.rnad_params_prev_,
    #   optimizers,
    #   timestep,
    #   alpha,
    #   update_net
    # )
     
    
    # v will not be used in future! Here it contains the regularized value function
    pi, v, _, _ = vectorized_net_apply(rnad_params, timestep.obs, timestep.legal)
    sim_pi = (pi - 0.5) * 2
    if self.config.similarity_metric == SimilarityMetric.POLICY_VALUE:
      similarity = jnp.concatenate((sim_pi, v), axis=-1)
      # similarity = jnp.concatenate((pi, v), axis=-1)
    elif self.config.similarity_metric == SimilarityMetric.POLICY:
      similarity = sim_pi
    elif self.config.similarity_metric == SimilarityMetric.VALUE:
      similarity = v
    elif self.config.similarity_metric == SimilarityMetric.LEGAL_ACTIONS:
      similarity = (timestep.legal * 2) - 1 # to be in range [-1, 1]
    elif self.config.similarity_metric == SimilarityMetric.LEGAL_POLICY_VALUE:
      # TODO: Legal actions have half of the weights that policy has.
      similarity = jnp.concatenate(((timestep.legal * 2) - 1, sim_pi, v), axis=-1)
    elif self.config.similarity_metric == SimilarityMetric.ACTION_HISTORY: 
      used_actions = jnp.tri(self.config.trajectory_max, self.config.trajectory_max, k=-1)
      action = (timestep.action[None, ...] - 0.5) * 0.5
      action = used_actions[..., None, None, None] * action
      action = jnp.moveaxis(action, 1, -2).reshape(*timestep.action.shape[:-1], -1) 
      similarity = action
    elif self.config.similarity_metric == SimilarityMetric.ACTION_HISTORY_POLICY: 
      
      # TODO: This can use Trajectory_max * trajectory_max - 1 instead, since the last action is never used
      used_actions = jnp.tri(self.config.trajectory_max, self.config.trajectory_max, k=-1)
      
      action = (timestep.action[None, ...] - 0.5) * 2
      action = used_actions[..., None, None, None] * action
      action = jnp.moveaxis(action, 1, -2).reshape(*timestep.action.shape[:-1], -1) 
      similarity = jnp.concatenate((action, sim_pi), axis=-1)
    elif self.config.similarity_metric == SimilarityMetric.ACTION_HISTORY_POLICY: 
      
      # TODO: This can use Trajectory_max * trajectory_max - 1 instead, since the last action is never used
      used_actions = jnp.tri(self.config.trajectory_max, self.config.trajectory_max, k=-1)
      
      action = (timestep.action[None, ...] - 0.5) * 0.5
      action = used_actions[..., None, None, None] * action
      action = jnp.moveaxis(action, 1, -2).reshape(*timestep.action.shape[:-1], -1) 
      similarity = jnp.concatenate((action, sim_pi), axis=-1)
      # similarity = jnp.concatenate((timestep.legal - 0.5, (pi * 2) - 1, v), axis=-1)
    elif self.config.similarity_metric == SimilarityMetric.ACTION_HISTORY_LEGAL:
      used_actions = jnp.tri(self.config.trajectory_max, self.config.trajectory_max, k=-1)
      action = (timestep.action[None, ...] - 0.5) * 0.5 
      action = used_actions[..., None, None, None] * action
      action = jnp.moveaxis(action, 1, -2).reshape(*timestep.action.shape[:-1], -1) 
      similarity = jnp.concatenate((action, (timestep.legal * 2) - 1), axis=-1)
    elif self.config.similarity_metric == SimilarityMetric.ACTION_HISTORY_LEGAL_POLICY:
      used_actions = jnp.tri(self.config.trajectory_max, self.config.trajectory_max, k=-1)
      action = (timestep.action[None, ...] - 0.5) * 0.5 
      action = used_actions[..., None, None, None] * action
      action = jnp.moveaxis(action, 1, -2).reshape(*timestep.action.shape[:-1], -1) 
      similarity = jnp.concatenate((action, (timestep.legal * 2) - 1, sim_pi), axis=-1)
    elif self.config.similarity_metric == SimilarityMetric.ISET_VECTOR:
      similarity = timestep.obs
    
    abstraction_params, ps_decoder_params, iset_encoder_params, similarity_params, optimizers, abstraction_loss = self.update_abstraction(
      network_parameters.abstraction_params,
      network_parameters.ps_decoder_params,
      network_parameters.iset_encoder_params,
      network_parameters.similarity_params,
      optimizers,
      similarity,
      timestep
    )
    
    transformation_params, optimizers, transformation_losses = self.update_transformations(
      network_parameters.transformation_params,
      abstraction_params,
      iset_encoder_params,
      optimizers,
      pi_before_train,
      pi,
      timestep
    )
    
    mvs_params, mvs_params_target, optimizers, mvs_loss = self.update_mvs(
      network_parameters.mvs_params,
      network_parameters.mvs_params_target,
      transformation_params,
      rnad_params,
      abstraction_params,
      iset_encoder_params,
      optimizers,
      timestep
    )
    
    
    legal_actions_params, optimizers, legal_loss = self.update_legal_actions(
      network_parameters.legal_actions_params,
      abstraction_params,
      iset_encoder_params,
      optimizers,
      timestep
    )
    
    dynamics_params, optimizers, dynamics_loss = self.update_dynamics(
      network_parameters.dynamics_params,
      abstraction_params,
      iset_encoder_params,
      optimizers,
      timestep)
    
    logs = {
      "Expected Loss": expected_loss,
      "RNaD Loss": rnad_loss,
      "P1 Abstraction Loss": abstraction_loss[0],
      "P2 Abstraction Loss": abstraction_loss[1],
      "P1 Transformation Loss": transformation_losses[0],
      "P2 Transformation Loss": transformation_losses[1],
      "MVS Loss": mvs_loss,
      "P1 Legal Actions Loss": legal_loss[0],
      "P2 Legal Actions Loss": legal_loss[1],
      "Dynamics Loss": dynamics_loss
      
    }
    
    return NetworkParameters(
      rnad_params=rnad_params,
      rnad_params_target=rnad_params_target,
      rnad_params_prev=rnad_params_prev,
      rnad_params_prev_=rnad_params_prev_,
      expected_params = expected_params,
      expected_params_target = expected_params_target,
      mvs_params=mvs_params,
      mvs_params_target=mvs_params_target,
      transformation_params=transformation_params,
      abstraction_params=abstraction_params,
      ps_decoder_params=ps_decoder_params,
      iset_encoder_params=iset_encoder_params,
      similarity_params=similarity_params,
      legal_actions_params = legal_actions_params,
      dynamics_params=dynamics_params
      ), optimizers, logs
  
  def step(self):
    trajectory = self.sample_trajectories()
    alpha, update_regularization = self._entropy_schedule(self.learner_steps)
    
    self.network_parameters, self.optimizers, logs = self.update_parameters(
      self.network_parameters,
      self.optimizers,
      trajectory,
      alpha, 
      update_regularization)
     
    # self.params, self.params_target, self.params_prev, self.params_prev_, self.optimizer, self.optimizer_target = self.update_parameters(
    #   self.params, self.params_target, self.params_prev, self.params_prev_, self.optimizer, self.optimizer_target, trajectory, alpha, update_regularization)
    
    self.learner_steps += 1
    
    
  @functools.partial(jax.jit, static_argnums=(0,))
  def update_jax_parameters(
    self,
    network_parameters: NetworkParameters,
    optimizers: Optimizers,
    key: chex.Array,
    alpha,
    update_net, 
  ):
    key = jax.random.split(key, self.config.batch_size)
    sample_trajectories = jax.vmap(self.sample_trajectory, in_axes=(None, 0), out_axes=1)
    trajectory = sample_trajectories(network_parameters.rnad_params, key)  
    
    return self.update_parameters(network_parameters, optimizers, lax.stop_gradient(trajectory), alpha, update_net)
   
  @functools.partial(jax.jit, static_argnums=(0, 4))
  def get_loss_value(
    self,
    network_parameters: NetworkParameters,
    optimizers: Optimizers,
    key: chex.Array,
    batch_size: int,
  ):
    key = jax.random.split(key, batch_size)
    sample_trajectories = jax.vmap(self.sample_trajectory, in_axes=(None, 0), out_axes=1)
    trajectory = sample_trajectories(network_parameters.rnad_params, key)  
    
    _, _, logs = self.update_parameters(network_parameters, optimizers, lax.stop_gradient(trajectory), 1.0, False)
    return logs

  
  def jax_step(self):
    key = self.get_next_rng_key()
    alpha, update_regularization = self._entropy_schedule(self.learner_steps)
    
    self.network_parameters, self.optimizers, logs = self.update_jax_parameters(
      self.network_parameters,
      self.optimizers,
      key, 
      alpha, 
      update_regularization)
    
    self.learner_steps +=1
    
    
  def multiple_jax_steps(self, iter: int):
    for _ in range(iter):
      self.jax_step() 
    
      
  def update_rnad_with_expected(
    self,
    rnad_params: Params,
    rnad_params_target: Params,
    rnad_params_prev: Params,
    rnad_params_prev_: Params,
    expected_params: Params,
    expected_params_target: Params,
    optimizers: Optimizers,
    timestep: TimeStep,
    alpha: float,
    update_net: bool
  ):
    if not self.config.train_rnad:
      return rnad_params, rnad_params_target, rnad_params_prev, rnad_params_prev_, optimizers, 0.0
    rnad_loss, rnad_grad = self._rnad_with_expected_loss(rnad_params, rnad_params_prev, rnad_params_prev_,  expected_params_target, timestep, alpha)
    
    rnad_params = optimizers.rnad_optimizer(rnad_params, rnad_grad)
    
    rnad_params_target = optimizers.rnad_optimizer_target(
        rnad_params_target, jax.tree.map(lambda a, b: a - b, rnad_params_target, rnad_params))
    
    rnad_params_prev, rnad_params_prev_ = jax.lax.cond(
        update_net,
        lambda: (rnad_params_target, rnad_params_prev),
        lambda: (rnad_params_prev, rnad_params_prev_))
     
    return rnad_params, rnad_params_target, rnad_params_prev, rnad_params_prev_, optimizers, rnad_loss
    
  def compute_q_values_from_expected(
    self,
    state_v: chex.Array,
    valid: chex.Array,
    sampling_policy: chex.Array,
    network_policy: chex.Array,
    regularization_term: chex.Array,
    action_oh: chex.Array,  
    reward: chex.Array, 
    eta: float = 0.2,
    gamma: float = 1.0
  ):
    # The reason we use this is to ensure this is weighted by the amount of the times we sample it
    importance_sampling = _policy_ratio(network_policy, sampling_policy, action_oh, valid)
    inverted_sampling = _policy_ratio(jnp.ones_like(sampling_policy), sampling_policy, action_oh, valid)
    
    # inverted_sampling = jnp.prod(inverted_sampling, axis=-2, keepdims=True)
    
    opponent_is = jnp.flip(importance_sampling, axis=-2)
    
    weighted_regularization_term = -eta * regularization_term
    regularization_entropy = eta * jnp.sum(network_policy * regularization_term, axis=-1)
    
    both_player_entropy = regularization_entropy[..., 1] - regularization_entropy[..., 0]
    
    both_player_entropy = jnp.stack((both_player_entropy, -both_player_entropy), axis=-1)
    
    # Should we use this or the usual reward o.O
    # opponent_regularized_reward = jnp.stack((reward, -reward), axis = -1) - jnp.flip(jnp.sum(action_oh * weighted_regularization_term, -1), -1) #-  jnp.flip(regularization_entropy, axis=-1) 
    
    opponent_regularized_reward = jnp.stack((reward, -reward), axis = -1) + jnp.flip(regularization_entropy, axis=-1)
    
    sampling_probability = jnp.sum(sampling_policy * action_oh, axis=-1, keepdims=True)
    sampling_probability = jnp.prod(sampling_probability, axis=-2, keepdims=True)
    
    network_reach_probability = jnp.sum(network_policy * action_oh, axis=-1, keepdims=True)
    counterfactual_reach = jnp.flip(network_reach_probability, -2)

    is_counterfactual_reach = counterfactual_reach / sampling_probability
    
    is_counterfactual_reach = jnp.concatenate((jnp.ones((1, *is_counterfactual_reach.shape[1:])), is_counterfactual_reach[:-1]), axis=0)
    is_counterfactual_reach = jnp.cumprod(is_counterfactual_reach, axis=0)
     
    state_v = jnp.stack((state_v, -state_v), axis=-2)
    
    counterfactual_value = state_v * is_counterfactual_reach
    
    next_state_v = jnp.concatenate((state_v[1:], jnp.zeros((1, *state_v.shape[1:]))), axis=0)
    
    # Is this necessary?
    state_v_without_entropy = state_v - jnp.expand_dims(regularization_entropy[..., (1, 0)], -1)
    
    q_value = state_v_without_entropy + weighted_regularization_term + action_oh * opponent_is * inverted_sampling * (opponent_regularized_reward[..., jnp.newaxis] + gamma * next_state_v - state_v_without_entropy)
    
    q_counterfactual_value = q_value * is_counterfactual_reach
    return counterfactual_value, q_counterfactual_value
    
    
  def rnad_with_expected_loss(
    self,
    rnad_params: Params, 
    rnad_params_prev: Params,
    rnad_params_prev_: Params, 
    expected_params: Params,
    timestep: TimeStep,
    alpha: float
  ):
    vectorized_net_apply = jax.vmap(jax.vmap(self.rnad_network.apply, in_axes=(None, 0, 0), out_axes=0), in_axes=(None, -2, -2), out_axes=-2)
    vectorized_expected_apply = jax.vmap(self.expected_network.apply, in_axes=(None, 0, 0), out_axes=0)
    
    pi, v, log_pi,  logit = vectorized_net_apply(rnad_params, timestep.obs, timestep.legal)
    _, _, log_pi_prev, _ = vectorized_net_apply(rnad_params_prev, timestep.obs, timestep.legal)
    _, _, log_pi_prev_, _ = vectorized_net_apply(rnad_params_prev_, timestep.obs, timestep.legal)
    
    
    state_v = vectorized_expected_apply(expected_params, timestep.obs[..., 0, :], timestep.obs[..., 1, :]) 
  
    expanded_valid = jnp.expand_dims(timestep.valid, (-2, -1))
    regularized_term = log_pi - (alpha * log_pi_prev + (1 - alpha) * log_pi_prev_)
    
    v_train_target, q_value = self.compute_q_values_from_expected(
      state_v, 
      expanded_valid, 
      timestep.policy, 
      pi, 
      regularized_term, 
      timestep.action, 
      timestep.reward,  
      eta=self.config.eta_regularization
    )
    v_loss = 0.0
    # We multiply by 2, since each player acts
    normalization = jnp.sum(timestep.valid) * 2 
    # v_loss = jnp.sum((expanded_valid * (v - lax.stop_gradient(v_train_target)) ** 2)) / (normalization + (normalization == 0))
     
    importance_sampling = jnp.ones_like(q_value)
    
    loss_neurd = neurd_loss(logit, pi, q_value, timestep.legal, importance_sampling)
    neurd_loss_value = -jnp.sum(loss_neurd * expanded_valid) / (normalization + (normalization == 0))
    
    return v_loss + neurd_loss_value
    
    
  def update_expected(
    self,
    expected_params: Params,
    expected_params_target: Params,
    rnad_params: Params,
    rnad_params_prev: Params,
    rnad_params_prev_: Params,
    optimizers: Optimizers,
    timestep: TimeStep,
    alpha: float
  ):
    if not self.config.train_rnad:
      return expected_params, expected_params_target, optimizers, 0.0
    expected_loss, expected_grad = self._expected_loss(expected_params, expected_params_target, rnad_params, rnad_params_prev, rnad_params_prev_, timestep, alpha)
    
    expected_params = optimizers.expected_optimizer(expected_params, expected_grad)
    
    expected_params_target = optimizers.expected_optimizer_target(
        expected_params_target, jax.tree.map(lambda a, b: a - b, expected_params_target, expected_params))
     
    
    return expected_params, expected_params_target, optimizers, expected_loss
    
    
  def expected_loss(self,
                    expected_params: Params,
                    expected_params_target: Params,
                    rnad_params: Params,
                    rnad_params_prev: Params,
                    rnad_params_prev_: Params,
                    timestep: TimeStep,
                    alpha: float):
    vectorized_net_apply = jax.vmap(jax.vmap(self.rnad_network.apply, in_axes=(None, 0, 0), out_axes=0), in_axes=(None, -2, -2), out_axes=-2)
    vectorized_expected_apply = jax.vmap(self.expected_network.apply, in_axes=(None, 0, 0), out_axes=0)
    
    pi, _, log_pi, _= vectorized_net_apply(rnad_params, timestep.obs, timestep.legal)
    
    _, _, log_pi_prev, _ = vectorized_net_apply(rnad_params_prev, timestep.obs, timestep.legal)
    _, _, log_pi_prev_, _ = vectorized_net_apply(rnad_params_prev_, timestep.obs, timestep.legal)
    
    
    v = vectorized_expected_apply(expected_params, timestep.obs[..., 0, :], timestep.obs[..., 1, :])
    v_target = vectorized_expected_apply(expected_params_target, timestep.obs[..., 0, :], timestep.obs[..., 1, :])
    
    expanded_valid = jnp.expand_dims(timestep.valid, (-1,))
    regularized_term = log_pi - (alpha * log_pi_prev + (1 - alpha) * log_pi_prev_)
    
    v_train_target = expected_v_trace(
      v_target, 
      expanded_valid,
      timestep.policy,
      pi,
      regularized_term,
      timestep.action,
      timestep.reward,
    )
    
    loss_v = expanded_valid * (v - lax.stop_gradient(v_train_target)) ** 2
    normalization = jnp.sum(timestep.valid)
    loss_v = jnp.sum(loss_v) / (normalization + (normalization == 0))
    return loss_v
    
  # Extract policy only in small games!
  def extract_full_policy(self):
    iset_set = []
    isets, legals = [], []
    def _traverse_tree(state: pyspiel.State):
      if state.is_terminal():
        return
      p1_iset = state.information_state_string(0)
      p2_iset = state.information_state_string(1)
      if not p1_iset in iset_set:
        iset_set.append(p1_iset)
        isets.append(state.information_state_tensor(0)) 
        legals.append(state.legal_actions_mask(0))
      if not p2_iset in iset_set:
        iset_set.append(p2_iset)
        isets.append(state.information_state_tensor(1)) 
        legals.append(state.legal_actions_mask(1))
      for a1 in state.legal_actions(0):
        for a2 in state.legal_actions(1):
          new_state = state.clone()
          new_state.apply_actions([a1, a2])
          _traverse_tree(new_state)
    _traverse_tree(self.game.new_initial_state())
    isets = np.array(isets, dtype=np.float32)
    legals = np.array(legals, dtype=np.int8)
    pi = self._jit_get_policy(self.network_parameters.rnad_params_target, isets, legals)
    policy = TabularPolicy(self.game)
    # policy.
    for i, iset in enumerate(iset_set):
      policy.action_probability_array[policy.state_lookup[iset]] = pi[i]
  
    return policy
  
  #WARNING!! This method will not extract full
  #policy for JAX games with chance nodes
  def extract_jax_policy(self, game):
    assert isinstance(self.game, JaxGame)
    iset_set = []
    isets, legals = [], []
    def _traverse_tree(state: pyspiel.State, game_state, key, current_legals, turn = 0):
      if state.is_terminal():
        return
      p1_iset = state.information_state_string(0)
      p2_iset = state.information_state_string(1)
      _, p1_jax_iset, p2_jax_iset, _ = self.game.get_info(game_state)
      if not p1_iset in iset_set:
        iset_set.append(p1_iset)
        isets.append(p1_jax_iset) 
        legals.append(current_legals[0])
      if not p2_iset in iset_set:
        iset_set.append(p2_iset)
        isets.append(p2_jax_iset)
        legals.append(current_legals[1])
      for a1 in state.legal_actions(0):
        for a2 in state.legal_actions(1):
          new_state = state.clone()
          new_state.apply_actions([a1, a2])
          key, subkey = jax.random.split(key)
          new_game_state, terminal, new_rewards, new_legals = self.game.apply_action(game_state, subkey, turn, np.array([a1, a2]))
          _traverse_tree(new_state, new_game_state, key, new_legals, turn + 1)

    key = jax.random.key(self.config.seed)    
    key, init_subkey, traverse_subkey = jax.random.split(key, 3)
    
    init_state, current_legals = self.game.initialize_structures(init_subkey)
    
    _traverse_tree(game.new_initial_state(), init_state, traverse_subkey, current_legals)
    isets = np.array(isets, dtype=np.float32)
    legals = np.array(legals, dtype=np.int8)
    pi = self._jit_get_policy(self.network_parameters.rnad_params_target, isets, legals)
    policy = TabularPolicy(game)
    # policy.
    for i, iset in enumerate(iset_set):
      policy.action_probability_array[policy.state_lookup[iset]] = pi[i]
  
    return policy
  
  def __getstate__(self):
    return dict(
      config=self.config,
      game = self.game, # TODO: If using pyspiel game, this probably breaks
      learner_steps = self.learner_steps,
      
      rngkey = self.rng_key,
      
      network_parameters = self.network_parameters,
      optimizers = jax.tree_map(lambda x: x.state, self.optimizers, is_leaf=lambda x: hasattr(x, 'state')), 
    )
    
  def __setstate__(self, state):
    self.config = state['config']
    self.game = state['game']
    self.init()
    
    self.learner_steps = state['learner_steps']
    
    self.rng_key = state['rngkey']
    
    self.network_parameters = state['network_parameters']
    # Can you do this better?
    self.optimizers.rnad_optimizer.state = state["optimizers"].rnad_optimizer
    self.optimizers.rnad_optimizer_target.state = state["optimizers"].rnad_optimizer_target
    self.optimizers.mvs_optimizer.state = state["optimizers"].mvs_optimizer
    self.optimizers.mvs_optimizer_target.state = state["optimizers"].mvs_optimizer_target
    self.optimizers.dynamics_optimizer.state = state["optimizers"].dynamics_optimizer
    self.optimizers.expected_optimizer.state = state["optimizers"].expected_optimizer
    self.optimizers.expected_optimizer_target.state = state["optimizers"].expected_optimizer_target
    for pl in range(2):
      self.optimizers.transformation_opitimizer[pl].state = state["optimizers"].transformation_opitimizer[pl]
      self.optimizers.abstraction_optimizer[pl].state = state["optimizers"].abstraction_optimizer[pl]
      self.optimizers.iset_encoder_optimizer[pl].state = state["optimizers"].iset_encoder_optimizer[pl]
      self.optimizers.similarity_optimizer[pl].state = state["optimizers"].similarity_optimizer[pl]
      self.optimizers.ps_decoder_optimizer[pl].state = state["optimizers"].ps_decoder_optimizer[pl]
      self.optimizers.legal_actions_optimizer[pl].state = state["optimizers"].legal_actions_optimizer[pl]
    
     

def main():
  
  from open_spiel.python.algorithms.mu_zero.jax_games.jax_goofspiel import JaxGoofspiel
  from open_spiel.python.algorithms.mu_zero.jax_games.jax_battleships import JaxBattleships
  cards = 5
  points_order = "descending"
  
  # params = {"num_cards": 5, "num_turns": 3, "first_round": 0}
  
  params = {"num_cards":cards, "points_order": points_order, "imp_info": True}
  
  orig_game =  pyspiel.load_game("goofspiel", params)
  
  
  game = JaxGoofspiel(cards, points_order)
  
  # board_shape = (2, 2)
  # ship_sizes = [2]
  # game = JaxBattleships(board_shape, ship_sizes)
  
  
  init, _ = game.initialize_structures(jax.random.key(0))
  
  _, init_p1, init_p2, init_ps = game.get_info(init)
  # game = orig_game 
  mu = True
  if mu == True:
    
    muzero = MuZeroTrain(game, MuZeroTrainConfig(batch_size=8, trajectory_max=game.max_trajectory_length(), use_abstraction=True, sampling_epsilon=0.0, entropy_schedule_size=(3000,), dynamics_type="public_state", similarity_metric="action_history_legal_policy"))
    
    # muzero = MuZeroTrain(game, MuZeroTrainConfig(batch_size=128, trajectory_max=cards - 1, use_abstraction=True, sampling_epsilon=0.0, entropy_schedule_size=(3000,), dynamics_type="public_state", similarity_metric="legal_actions"))
    # muzero.rng_key = jax.random.PRNGKey(42)
  else:
    params = [(n, p) for n, p in params.items()]
    muzero = RNaDSolver(RNaDConfig(game_name="goofspiel", game_params=params, trajectory_max=cards-1, batch_size=32))
  
  
  # profiler = Profiler()
  # profiler.start()
  with chex.fake_jit():
    for _ in range(2000):
      muzero.jax_step()
  print("Trained")
      
     
  policy = muzero.extract_jax_policy(orig_game) 
  
if __name__ == "__main__":
  main()