
from open_spiel.python.algorithms.rnad.rnad import _legal_policy, legal_log_policy, EntropySchedule
from open_spiel.python.algorithms.mu_zero.flax_utils import init_network_with_optimizer, init_params_optimizer, optax_optimizer
from open_spiel.python.algorithms.mu_zero.sim_rnad import RNaDNework

from open_spiel.python.algorithms.rnad.rnad import RNaDSolver, RNaDConfig
from open_spiel.python.policy import TabularPolicy
from open_spiel.python.algorithms.exploitability import exploitability

from typing import Sequence, Any, Callable
from pyinstrument import Profiler
import jax
import jax.numpy as jnp
import jax.lax as lax

import flax.linen as nn
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
  iset_encoder_optimizer: Sequence[Optimizer]  = ()
  similarity_optimizer: Sequence[Optimizer]  = ()
  dynamics_optimizer: Optimizer = ()
 
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
  iset_encoder_params: Sequence[Params] = ()
  similarity_params: Sequence[Params] = ()
  dynamics_params: Params = ()
  
  
  
# Holds the important properties that should be important as a similarity metric between 2 infosets for clustering
class SimilarityNetwork(nn.Module):
  hidden_size: int
  out_dim: int
  
  
  @nn.compact
  def __call__(self, x: chex.Array) -> Any:
    x = nn.Dense(self.hidden_size)(x)
    x = nn.relu(x)
    x = nn.Dense(self.hidden_size)(x)
    x = nn.relu(x)
    x = nn.Dense(self.out_dim)(x)
    return x

# TODO: This may be better to output public state and indices of isets, but who knows
class DynamicsNetwork(nn.Module):
  hidden_size: int
  abstraction_size: int
  
  @nn.compact
  def __call__(self, p1_isets, p2_isets, p1_action, p2_action):
    x = jnp.concatenate((p1_isets, p2_isets, p1_action, p2_action), axis=-1)
    x = nn.Dense(self.hidden_size)(x)
    x = nn.relu(x)
    x = nn.Dense(self.hidden_size)(x)
    x = nn.relu(x)
    
    next_p1_iset = nn.Dense(self.abstraction_size)(x)
    next_p2_iset = nn.Dense(self.abstraction_size)(x)
    reward = nn.Dense(2)(x)
    is_terminal = nn.Dense(1)(x)
    
    return next_p1_iset, next_p2_iset, reward, is_terminal
 
class InfosetEncoder(nn.Module):
  hidden_size: int
  isets: int
  
  @nn.compact
  def __call__(self, x: chex.Array) -> Any:
    x = nn.Dense(self.hidden_size)(x)
    x = nn.relu(x)
    x = nn.Dense(self.hidden_size)(x)
    x = nn.relu(x)
    iset = nn.Dense(self.isets)(x)
    return iset 

class PublicStateEncoder(nn.Module):
  hidden_size: int
  iset_size: int
  isets: int
  
  @nn.compact
  def __call__(self, x: chex.Array) -> chex.Array:
    x = nn.Dense(self.hidden_size)(x)
    x = nn.relu(x)
    x = nn.Dense(self.hidden_size)(x)
    x = nn.relu(x)
    ps = nn.Dense(self.iset_size * self.isets)(x)
    ps = jnp.squeeze(ps.reshape((-1, self.isets, self.iset_size)))
    return ps


class LegalActionsNetwork(nn.Module):
  hidden_size: int
  out_dim: int
  
  
  @nn.compact
  def __call__(self, x: chex.Array) -> chex.Array:
    x = nn.Dense(self.hidden_size)(x)
    x = nn.relu(x)
    x = nn.Dense(self.hidden_size)(x)
    x = nn.relu(x)
    x = nn.Dense(self.out_dim)(x)
    # x = nn.sigmoid(x)
    return x
 
class TransformationNetwork(nn.Module):
  hidden_size: int
  transformations: int
  actions: int
  
  @nn.compact
  def __call__(self, x: chex.Array) -> chex.Array:
    x = nn.Dense(self.hidden_size)(x)
    x = nn.relu(x)
    x = nn.Dense(self.hidden_size)(x)
    x = nn.relu(x)
    x = nn.Dense(self.transformations * self.actions)(x)
    x = jnp.squeeze(x.reshape((-1, self.transformations, self.actions)))
    return x

class MAVSNetwork(nn.Module):
  hidden_size: int
  values: int 
  
  @nn.compact
  def __call__(self, p1_iset: chex.Array, p2_iset: chex.Array) -> chex.Array:
    x = jnp.concatenate((p1_iset, p2_iset), axis=-1)
    x = nn.Dense(self.hidden_size)(x)
    x = nn.relu(x)
    x = nn.Dense(self.hidden_size)(x)
    x = nn.relu(x)
    x = nn.Dense(self.values * self.values)(x)
    x = jnp.squeeze(x.reshape((-1, self.values, self.values)))
    return x 
  
class MUVSNetwork(nn.Module):
  hidden_size: int
  p1_values: int
  p2_values: int
  
  @nn.compact
  def __call__(self, p1_iset: chex.Array, p2_iset: chex.Array) -> chex.Array:
    x = jnp.concatenate((p1_iset, p2_iset), axis=-1)
    x = nn.Dense(self.hidden_size)(x)
    x = nn.relu(x)
    x = nn.Dense(self.hidden_size)(x)
    x = nn.relu(x)
    x = nn.Dense(self.values + self.values)(x)
    x = jnp.squeeze(x.reshape((-1, 2, self.values)))
    return x
 
class ExpectedNetwork(nn.Module):
  hidden_size: int 
  
  
  @nn.compact
  def __call__(self, p1_iset: chex.Array, p2_iset: chex.Array) -> chex.Array:
    x = jnp.concatenate((p1_iset, p2_iset), axis=-1)
    x = nn.Dense(self.hidden_size)(x)
    x = nn.relu(x)
    x = nn.Dense(self.hidden_size)(x)
    x = nn.relu(x)
    x = nn.Dense(1)(x)
    return x
 
class SimilarityMetric(str, Enum):
  POLICY = "policy"
  VALUE = "value"
  POLICY_VALUE = "policy_value"
  LEGAL_ACTIONS = "legal_actions"
 
@chex.dataclass(frozen=True)
class MuZeroConfig: 
  
  batch_size: int = 32
  
  trajectory_max: int = 6
  sampling_epsilon: float = 0.0
  
  train_rnad: bool = True
  train_mvs: bool = True
  train_abstraction: bool = True
  train_dynamics: bool = True
  
  
  use_abstraction: bool = False
  abstraction_amount: int = 10
  abstraction_size: int = 32
  similarity_metric: SimilarityMetric = SimilarityMetric.POLICY_VALUE
  
  ps_hidden_size: int = 128
  iset_hidden_size: int = 64
  dynamics_hidden_size: int = 64
  legal_actions_hidden_size: int = 64
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
  

def _policy_ratio(pi: chex.Array, mu: chex.Array, actions_oh: chex.Array, valid: chex.Array) -> chex.Array: 
  pi_actions_prob = jnp.sum(pi * actions_oh, axis=-1, keepdims=True) * valid + (1 - valid)
  mu_actions_prob = jnp.sum(mu * actions_oh, axis=-1, keepdims=True) * valid + (1 - valid)
  
  return pi_actions_prob / mu_actions_prob

  

def tree_where(pred: chex.Array, x: chex.ArrayTree, y: chex.ArrayTree) -> chex.ArrayTree:
  
  def _where(x, y):
    return jnp.where(pred, x, y)
  
  return jax.tree.map(_where, x, y)
  
def apply_force_with_threshold(decision_outputs: chex.Array, force: chex.Array,
                               threshold: float,
                               threshold_center: chex.Array) -> chex.Array:
  """Apply the force with below a given threshold."""
  chex.assert_equal_shape((decision_outputs, force, threshold_center))
  can_decrease = decision_outputs - threshold_center > -threshold
  can_increase = decision_outputs - threshold_center < threshold
  force_negative = jnp.minimum(force, 0.0)
  force_positive = jnp.maximum(force, 0.0)
  clipped_force = can_decrease * force_negative + can_increase * force_positive
  return decision_outputs * lax.stop_gradient(clipped_force)



def neurd_loss(
  logits: chex.Array,
  policy: chex.Array,
  q_values: chex.Array, 
  legal: chex.Array,
  importance_sampling: chex.Array,
  clip: float=10_000,
  threshold: float=2.0
):
  advantage = q_values - jnp.sum(policy * q_values, axis=-1, keepdims=True)
  advantage = advantage * importance_sampling
  advantage = lax.stop_gradient(jnp.clip(advantage, -clip, clip))
  mean_logit = jnp.sum(logits * legal, axis=-1, keepdims=True) / jnp.sum(legal, axis=-1, keepdims=True)
  
  logits_shifted = logits - mean_logit
  threshold_ceter = jnp.zeros_like(logits_shifted)
  
  neurd_loss_value = jnp.sum(legal * apply_force_with_threshold(logits_shifted, advantage, threshold, threshold_ceter), axis=-1, keepdims=True)
  
  return neurd_loss_value

# TODO: Verify that merges the vectors corectly
def transform_trajectory_to_last_dimension(x: chex.Array) -> chex.Array:
  return jnp.moveaxis(x, 0, -2).reshape((*x.shape[1:-1], -1))

def normalize_direction_with_mask(x:chex.Array, mask:chex.Array) -> chex.Array:

  x = mask * x 
  norm = jnp.linalg.norm(x, 2, -1, keepdims=True)
  return jnp.where(norm < 1e-15, x, x / norm)

# TODO: This should take valid into account
def _compute_soft_kmeans_loss_with_cluster_assignments(real:chex.Array, pred: chex.Array, temperature: float=1.0):
  # The predicted dimension is missing
  chex.assert_shape((real,), (*pred.shape[:-2], pred.shape[-1]))
  cluster_difference = lax.stop_gradient(jnp.expand_dims(real, -2)) - pred
  cluster_distance = jnp.linalg.norm(cluster_difference, axis=-1)
  cluster_soft_assignement = jax.nn.softmax(-cluster_distance * temperature, axis=-1)
  cluster_loss = jnp.sum(cluster_difference ** 2, axis=-1)
  cluster_loss = jnp.sum(cluster_loss * cluster_soft_assignement, axis=-1)
  return jnp.mean(cluster_loss), cluster_soft_assignement
  
def _compute_soft_kmeans_loss_with_single(real, pred, probs):
  cluster_loss, cluster_soft_assignement = _compute_soft_kmeans_loss_with_cluster_assignments(real, pred, 1.0)
  prob_loss = optax.losses.softmax_cross_entropy(probs,  jax.lax.stop_gradient(cluster_soft_assignement))
  # labels = jnp.argmax(cluster_soft_assignement, axis=-1)
  # smoothed_labels = jax.nn.one_hot(jnp.argmax(probs, axis=-1), probs.shape[-1])
  # smoothed_labels = optax.smooth_labels(smoothed_labels, 0.1)
  
  # prob_loss = optax.losses.softmax_cross_entropy_with_integer_labels(probs, jax.lax.stop_gradient(jnp.argmax(cluster_soft_assignement, axis=-1)))
  return cluster_loss + jnp.mean(prob_loss)

# This contains RNaD implementation. Note that this implementation is specific for two-player zero-sum games. Unlike the open_spiel RNaD that can be used to general-sum multiplayer games.
class MuZero():
  def __init__(self, game, config) -> None:
    assert config.matrix_valued_states, "Multi-valued states are not implemented."
    self.config = config
    self.game = game
    
    if isinstance(self.game, JaxOriginalGoofspiel):
      print("Warning: you use Jax Goofspiel, so you need to use domain specific goofspiel_step method")
      
    self.init()
    
  def init(self):
    self.actions = self.game.num_distinct_actions()
    
    if self.config.use_abstraction:
      self.obs = self.config.abstraction_size
    else:
      self.obs = self.game.information_state_tensor_shape()
      
    self.rng_key = jax.random.PRNGKey(self.config.seed)
    
    # temp_keys = self.get_next_rng_keys(6)
    
    self.example_state  = self.game.new_initial_state()
    self.example_timestep = self.default_timestep()
    self.example_obs = np.ones((1, self.obs))
    
    self._entropy_schedule = EntropySchedule(
        sizes=self.config.entropy_schedule_size,
        repeats=self.config.entropy_schedule_repeats)
    
    self.rnad_network = RNaDNework(self.config.rnad_hidden_size, self.actions)
    self.abstraction_network = PublicStateEncoder(self.config.ps_hidden_size, self.config.abstraction_size, self.config.abstraction_amount)
    self.iset_encoder = InfosetEncoder(self.config.iset_hidden_size, self.config.abstraction_amount)
    self.similarity_network = SimilarityNetwork(self.config.iset_hidden_size, self.similarity_output_size())
    # self.dynamics_network = DynamicsNetwork(self.config.dynamics_hidden_size, self.example_timestep.obs.shape[-1])
    self.dynamics_network = DynamicsNetwork(self.config.dynamics_hidden_size, self.obs)
    
    self.transformation_network = TransformationNetwork(self.config.dynamics_hidden_size, self.config.transformations, self.actions)
    self.mvs_network = MAVSNetwork(self.config.dynamics_hidden_size, self.config.transformations + 1)
    
    
    self._rnad_loss = jax.value_and_grad(self.rnad_loss, has_aux=False)
    self._abstraction_loss = jax.value_and_grad(self.abstraction_loss, argnums=[0,1,2], has_aux=False)
    if self.config.use_abstraction:
      self._dynamics_loss = jax.value_and_grad(self.abstracted_dynamics_loss, has_aux=False)
      self._transformation_loss = jax.value_and_grad(self.abstracted_transformation_loss, has_aux=False)
      self._mvs_loss = jax.value_and_grad(self.abstracted_mvs_loss, has_aux=False)
    else: 
      self._dynamics_loss = jax.value_and_grad(self.non_abstracted_dynamics_loss, has_aux=False)
      self._transformation_loss = jax.value_and_grad(self.non_abstracted_transformation_loss, has_aux=False)
      self._mvs_loss = jax.value_and_grad(self.non_abstracted_mvs_loss, has_aux=False)
    
    # self._mvs_loss = 
    
    # self._dynamics_loss = jax.value_and_grad(self.non_abstracted_dynamics_loss, has_aux=False)
    # self._non_abstracted_dynamics_loss = jax.value_and_grad(self.non_abstracted_dynamics_loss, has_aux=False)
    
    temp_key = self.get_next_rng_key()
    params = self.rnad_network.init(temp_key, self.example_timestep.obs, self.example_timestep.legal)
    params_target = self.rnad_network.init(temp_key, self.example_timestep.obs, self.example_timestep.legal)
    params_prev = self.rnad_network.init(temp_key, self.example_timestep.obs, self.example_timestep.legal)
    params_prev_ = self.rnad_network.init(temp_key, self.example_timestep.obs, self.example_timestep.legal)
    
    optimizer = optax_optimizer(params, optax.chain(optax.adam(self.config.learning_rate, b1=0.0), optax.clip(100)))
    optimizer_target = optax_optimizer(params_target, optax.sgd(self.config.target_network_update))
    
    temp_keys = self.get_next_rng_keys(11)
    
    
    # TODO: Different init?
    p1_abstraction_params = self.abstraction_network.init(temp_keys[0], self.example_timestep.public_state)
    p2_abstraction_params = self.abstraction_network.init(temp_keys[1], self.example_timestep.public_state)
    # TODO: Do we want 2 different networks for iset encoder and legal action?
    p1_iset_encoder_params = self.iset_encoder.init(temp_keys[2], self.example_timestep.obs)
    p2_iset_encoder_params = self.iset_encoder.init(temp_keys[3], self.example_timestep.obs)
    
    # Similarity always uses abstraction
    p1_similarity_params = self.similarity_network.init(temp_keys[4], np.ones((1, self.config.abstraction_size)))
    p2_similarity_params = self.similarity_network.init(temp_keys[5], np.ones((1, self.config.abstraction_size)))
    
    # self.dynamics_params = self.dynamics_network.init(temp_keys[6], self.example_timestep.obs, self.example_timestep.obs, self.example_timestep.action, self.example_timestep.action)
    dynamics_params = self.dynamics_network.init(temp_keys[6], self.example_obs, self.example_obs, self.example_timestep.action, self.example_timestep.action)
    
    mvs_params = self.mvs_network.init(temp_keys[7], self.example_obs, self.example_obs)
    mvs_params_target = self.mvs_network.init(temp_keys[7], self.example_obs, self.example_obs)
    
    p1_transformation_params = self.transformation_network.init(temp_keys[8], self.example_obs)
    p2_transformation_params = self.transformation_network.init(temp_keys[9], self.example_obs)
    
    
    p1_abstraction_optimizer = optax_optimizer(p1_abstraction_params, optax.chain(optax.adam(self.config.learning_rate), optax.clip(100)))
    p2_abstraction_optimizer = optax_optimizer(p2_abstraction_params, optax.chain(optax.adam(self.config.learning_rate), optax.clip(100)))
    p1_iset_encoder_optimizer = optax_optimizer(p1_iset_encoder_params, optax.chain(optax.adam(self.config.learning_rate), optax.clip(100)))
    p2_iset_encoder_optimizer = optax_optimizer(p2_iset_encoder_params, optax.chain(optax.adam(self.config.learning_rate), optax.clip(100)))
    p1_similarity_optimizer = optax_optimizer(p1_similarity_params, optax.chain(optax.adam(self.config.learning_rate), optax.clip(100)))
    p2_similarity_optimizer = optax_optimizer(p2_similarity_params, optax.chain(optax.adam(self.config.learning_rate), optax.clip(100)))
    
    dynamics_optimizer = optax_optimizer(dynamics_params, optax.chain(optax.adam(self.config.learning_rate), optax.clip(100)))
    
    mvs_optimizer = optax_optimizer(mvs_params, optax.chain(optax.adam(self.config.learning_rate), optax.clip(100)))
    mvs_optimizer_target = optax_optimizer(mvs_params_target, optax.sgd(self.config.target_network_update))
    
    p1_transformation_optimizer = optax_optimizer(p1_transformation_params, optax.chain(optax.adam(self.config.learning_rate), optax.clip(100)))
    p2_transformation_optimizer = optax_optimizer(p2_transformation_params, optax.chain(optax.adam(self.config.learning_rate), optax.clip(100)))
    
    self.expected_network = ExpectedNetwork(self.config.rnad_hidden_size)
    
    expected_params = self.expected_network.init(temp_keys[10], self.example_timestep.obs, self.example_timestep.obs)
    expected_params_target = self.expected_network.init(temp_keys[10], self.example_timestep.obs, self.example_timestep.obs)
    
    expected_optimizer = optax_optimizer(expected_params, optax.chain(optax.adam(self.config.learning_rate), optax.clip(100)))
    expected_optimizer_target = optax_optimizer(expected_params_target, optax.sgd(self.config.target_network_update))
    
    self._expected_loss = jax.value_and_grad(self.expected_loss, has_aux=False)
    
    self._rnad_with_expected_loss = jax.value_and_grad(self.rnad_with_expected_loss, has_aux=False)
    
    self.optimizers = Optimizers(
      rnad_optimizer = optimizer,
      rnad_optimizer_target = optimizer_target,
      expected_optimizer = expected_optimizer,
      expected_optimizer_target = expected_optimizer_target,
      mvs_optimizer = mvs_optimizer,
      mvs_optimizer_target = mvs_optimizer_target,
      transformation_opitimizer = (p1_transformation_optimizer, p2_transformation_optimizer),
      abstraction_optimizer = (p1_abstraction_optimizer, p2_abstraction_optimizer),
      iset_encoder_optimizer = (p1_iset_encoder_optimizer, p2_iset_encoder_optimizer),
      similarity_optimizer = (p1_similarity_optimizer, p2_similarity_optimizer),
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
      iset_encoder_params = (p1_iset_encoder_params, p2_iset_encoder_params),
      similarity_params = (p1_similarity_params, p1_similarity_params),
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
    assert False, "Unknown similarity metric"   

  def default_timestep(self):
    obs = np.zeros(self.game.information_state_tensor_shape(), dtype=np.float32)
    public_state = np.zeros(self.game.public_state_tensor_shape(), dtype=np.float32)
    
    legal = np.ones(self.actions, dtype=np.int8)
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
    return jax.tree_util.tree_map(lambda xs: np.expand_dims(xs, 0), ts)
    

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
  def _jit_get_legal_actions(self, abstract_params, iset_encoder_params, legal_params, public_state, obs) -> chex.Array:
    isets = self.abstraction_network.apply(abstract_params, public_state)
    iset_probs = self.iset_encoder.apply(iset_encoder_params, obs)
    iset_choice = jnp.argmax(iset_probs, axis=-1)
    legal = self.legal_actions_network.apply(legal_params, isets[jnp.arange(public_state.shape[0]), iset_choice])
    return legal
    
  @functools.partial(jax.jit, static_argnums=(0,))
  def _jit_get_next_state(self, params, p1_iset, p2_iset, p1_action, p2_action):
    p1_action = jax.nn.one_hot(p1_action, self.actions)
    p2_action = jax.nn.one_hot(p2_action, self.actions)
    return self.dynamics_network.apply(params, p1_iset, p2_iset, p1_action, p2_action)
  
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
    return jnp.squeeze(jnp.take_along_axis(abstraction, picked_iset[..., jnp.newaxis], axis=-2))
  
  @functools.partial(jax.jit, static_argnums=(0, ))
  def _jit_get_mvs(self, mvs_params, p1_iset, p2_iset):
    return self.mvs_network.apply(mvs_params, p1_iset, p2_iset)
  
  # The observaiton is already only for a given player pl
  def get_abstraction(self, public_state, obs, pl):
    return self._jit_get_abstraction(self.network_parameters.abstraction_params[pl], self.network_parameters.iset_encoder_params[pl], public_state, obs) 
  
  def get_both_abstraction(self, public_state, p1_iset, p2_iset):
    p1_abstraction_iset = self._jit_get_abstraction(self.network_parameters.abstraction_params[0], self.network_parameters.iset_encoder_params[0], public_state, p1_iset)
    p2_abstraction_iset = self._jit_get_abstraction(self.network_parameters.abstraction_params[1], self.network_parameters.iset_encoder_params[1], public_state, p2_iset)
    return p1_abstraction_iset, p2_abstraction_iset
  
  def get_non_abstracted_next_state(self, public_state, p1_iset, p2_iset, p1_action, p2_action):
    return self._jit_get_next_state(self.network_parameters.dynamics_params, p1_iset, p2_iset, p1_action, p2_action)
  
  # Expects isets in the original game definition and action as a index of the action
  def get_next_state(self, public_state, p1_iset, p2_iset, p1_action, p2_action):
    if self.config.use_abstraction:
      p1_iset, p2_iset = self.get_both_abstraction(public_state, p1_iset, p2_iset)
    return self._jit_get_next_state(self.network_parameters.dynamics_params, p1_iset, p2_iset, p1_action, p2_action) 
    
  def get_legal_actions(self, public_state, obs, pl): 
    assert False, "Do not call this method, it is here only because of the older PoC version."
    return self._jit_get_legal_actions(self.network_parameters.abstraction_params[pl], self.network_parameters.iset_encoder_params[pl], self.network_parameters.legal_params[pl], public_state, obs)
  
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
  
  
  @functools.partial(jax.jit, static_argnums=(0,))
  def sample_goofspiel_trajectories(self, params, key) -> TimeStep:
    key = jax.random.split(key, (self.config.trajectory_max, self.config.batch_size, 2))
    # turns = list(range(self.game.cards -1))
    cards = self.game.cards
    
    @chex.dataclass(frozen=True)
    class SampleTrajectoryCarry:
      point_cards: chex.Array
      played_cards: chex.Array
      p1_points: chex.Array
      legal_actions: chex.Array
      
    point_cards, played_cards, p1_points, legal_actions = self.game.initialize_batch_structures(self.config.batch_size)
    
    @jax.jit
    def choice_wrapper(key, p):
      action = jax.random.choice(key, cards, p=p)
      action_oh = jax.nn.one_hot(action, cards)
      return action, action_oh
    
    vectorized_get_info = jax.vmap(self.game.get_info, in_axes=(0, 0, 0), out_axes=(0, 0, 0, 0))
    vectorized_apply_action = jax.vmap(self.game.apply_action, in_axes=(0, 0, 0, None, 0), out_axes=(0, 0, 0, 0, 0))
    vectorized_sample_action = jax.vmap(jax.vmap(choice_wrapper, in_axes=(0, 0), out_axes=0), in_axes=(0, 0), out_axes=0)
    network_apply = jax.vmap(self._jit_get_policy, in_axes=(None, 1, 1), out_axes=1)
    
    init_carry = SampleTrajectoryCarry(
      point_cards=point_cards,
      played_cards=played_cards,
      p1_points=p1_points,
      legal_actions=legal_actions
    )
    
    def _sample_trajectory(carry: SampleTrajectoryCarry, xs) -> tuple[SampleTrajectoryCarry, chex.Array]:
      (key, turn) = xs
      _, p1_iset, p2_iset, public_state = vectorized_get_info(carry.point_cards, carry.played_cards, carry.p1_points)
      obs = jnp.stack((p1_iset, p2_iset), axis=1)
      pi = network_apply(params, obs, carry.legal_actions)
      random_pi = carry.legal_actions / jnp.sum(carry.legal_actions, axis=-1, keepdims=True)
      pi = self.config.sampling_epsilon * random_pi + (1 - self.config.sampling_epsilon) * pi
      # pi = carry.legal_actions / jnp.sum(carry.legal_actions, axis=-1, keepdims=True)
      action, action_oh = vectorized_sample_action(key, pi)
      next_legal, next_rewards, next_point_cards, next_played_cards, next_p1_points = vectorized_apply_action(carry.point_cards, carry.played_cards, carry.p1_points, turn, action)
      new_carry = SampleTrajectoryCarry(
        point_cards=next_point_cards,
        played_cards=next_played_cards,
        p1_points=next_p1_points,
        legal_actions=next_legal
      )
      timestep = TimeStep(
        valid = jnp.ones_like(next_rewards),
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
             xs=(key, jnp.arange(cards - 1)))
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
    
  def v_trace(self, 
    v: chex.Array,
    valid: chex.Array,
    sampling_policy: chex.Array,
    network_policy: chex.Array,
    regularization_term: chex.Array,
    action_oh: chex.Array,
    reward: chex.Array, # Still not regularized
    lambda_: float = 1.0, # Lambda parameter for V-trace
    c: float = 1.0, # Importance sampling clipping
    rho: float = np.inf, # Importance sampling clipping
    eta: float = 0.2, # Regularization factor 
    gamma: float = 1.0 # Discount factor
  ):
    importance_sampling = _policy_ratio(network_policy, sampling_policy, action_oh, valid)
    
    # The reason we use this is to ensure this is weighted by the amount of the times we sample it
    inverted_sampling = _policy_ratio(jnp.ones_like(sampling_policy), sampling_policy, action_oh, valid)
    
    regularization_entropy = eta * jnp.sum(network_policy * regularization_term, axis=-1)
    weighted_regularization_term = -eta * regularization_term# + regularization_entropy[..., (1, 0), jnp.newaxis]
    
    both_player_entropy = (regularization_entropy[..., 1] - regularization_entropy[..., 0])
    
    entropy_reward = reward + both_player_entropy
    entropy_reward = jnp.expand_dims(jnp.stack((entropy_reward, -entropy_reward), axis=-1), -1)
    
    
    q_reward = jnp.stack((reward, -reward), axis=-1) + regularization_entropy[..., (1, 0)]
    
    q_reward = jnp.expand_dims(q_reward, -1)
    
    
    
    @chex.dataclass(frozen=True)
    class VTraceCarry: 
      next_value: chex.Array # Network value in the next timestep 
      delta_v: chex.Array # Propagated delta V in V-trace from the next timestep
    
    
    init_carry = VTraceCarry(
      next_value=jnp.zeros_like(v[-1]),
      delta_v=jnp.zeros_like(v[-1])
    )
  
    def _v_trace(carry: VTraceCarry, x) -> tuple[VTraceCarry, Any]:
      (importance_sampling, v, q_reward, entropy_reward, weighted_regularization_term, valid, inverted_sampling, action_oh) = x 
      # reward_uncorrected = reward + gamma * carry.reward_uncorrected + entropy
      # discounted_reward = reward + gamma * carry.reward
      
      delta_v = jnp.minimum(rho, importance_sampling) * (entropy_reward + gamma * carry.next_value - v)
      carry_delta_v = delta_v + lambda_ * jnp.minimum(c, importance_sampling) * gamma * carry.delta_v
      
      v_target = v + carry_delta_v
      
      # TODO: Shall we use opponent entropy reg term or entropy of played action?
      
      # We use importance sampling of the opponent.
      opponent_sampling = jnp.flip(importance_sampling, -2)
      
      q_value = v + weighted_regularization_term  + action_oh * opponent_sampling * inverted_sampling  * (q_reward + gamma * (carry.next_value + carry.delta_v) - v )
      
      
      # q_value = weighted_regularization_term + action_oh * opponent_sampling * inverted_sampling * (q_reward + gamma * (carry.next_value + carry.delta_v))
      
      next_carry = VTraceCarry(
        next_value=v,
        delta_v=carry_delta_v
      )
      reset_carry = init_carry
    
      reset_v_target = jnp.zeros_like(v_target)
      reset_q_value = jnp.zeros_like(q_value) 
      
      reset_carry = init_carry
      return tree_where(valid, (next_carry, (v_target, q_value)), (reset_carry, (reset_v_target, reset_q_value)))
      # return jnp.where(valid, next_carry, reset_carry), (v_target, q_value)
      
      
      
    _, (v_target, q_value) = lax.scan(
      f=_v_trace,
      init=init_carry,
      xs=(importance_sampling, v, q_reward, entropy_reward, weighted_regularization_term, valid, inverted_sampling, action_oh),
      reverse=True
    ) 
    return v_target, q_value
  
  
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
    
    v_train_target, q_value = self.v_trace(v_target, expanded_valid, timestep.policy, pi, regularized_term, timestep.action, timestep.reward, c=self.config.c_iset_vtrace, rho=self.config.rho_iset_vtrace, eta=self.config.eta_regularization)
    
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
    
    loss, _ = _compute_soft_kmeans_loss_with_cluster_assignments(update_direction, predicted_direction)
    return loss
  
  def state_v_trace(
    self,
    v: chex.Array,
    sampling_policy: chex.Array,
    transformed_policy: chex.Array, 
    actions_oh: chex.Array,
    valid: chex.Array,
    reward: chex.Array, # Still not regularized
    lambda_: float = 1.0, # Lambda parameter for V-trace
    c: float = 1.0, # Importance sampling clipping
    rho: float = 1.0, # Importance sampling clipping 
    gamma: float = 1.0 # Discount factor
  ) -> chex.Array:
    pi_action_prob = jnp.sum(transformed_policy * jnp.expand_dims(actions_oh, -3), axis=-1)
    mu_action_prob = jnp.sum(sampling_policy * actions_oh, axis=-1)
    importance_sampling = pi_action_prob / jnp.expand_dims(mu_action_prob, -2)
    
    p1_is = importance_sampling[..., 0, None]
    p2_is = jnp.expand_dims(importance_sampling[..., 1], -2)
    @chex.dataclass(frozen=True)
    class StateVTraceCarry:
      """The carry of the v-trace scan loop."""
      next_state_value: chex.Array
      next_state_delta_v: chex.Array
      
    init_carry = StateVTraceCarry(
      next_state_value=jnp.zeros_like(v[-1]),
      next_state_delta_v=jnp.zeros_like(v[-1])
      
    )
    def _state_v_trace(carry: StateVTraceCarry, x) -> tuple[StateVTraceCarry, Any]:
      (p1_is, p2_is, v, reward, valid) = x
      
      delta_v = jnp.minimum(rho, p1_is) * jnp.minimum(rho, p2_is) * (reward + gamma * carry.next_state_value - v)
      
      carry_delta_v = delta_v + lambda_ * jnp.minimum(c, p1_is) * jnp.minimum(c, p2_is) * gamma * carry.next_state_delta_v
      
      v_target = v + carry_delta_v
      
      reset_carry = init_carry
      next_carry = StateVTraceCarry(
        next_state_value=v,
        next_state_delta_v=carry_delta_v
      )
      return tree_where(valid, (next_carry, v_target), (reset_carry, jnp.zeros_like(v_target)))
    
    _, v_target = lax.scan(
      f=_state_v_trace,
      init=init_carry,
      xs=(p1_is, p2_is, v, jnp.expand_dims(reward, (-1, -2)), jnp.expand_dims(valid, (-1, -2))),
      reverse=True
    )
    
    return v_target
  
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
     
    mvs_train_target = self.state_v_trace(mvs_target, timestep.policy, policy_transformations, timestep.action, timestep.valid, timestep.reward, c=self.config.c_state_vtrace, rho=self.config.rho_state_vtrace)
    
    # mask = timestep.valid[..., jnp.newaxis, jnp.newaxis]

    loss_v = timestep.valid[..., jnp.newaxis, jnp.newaxis] * (mvs - lax.stop_gradient(mvs_train_target)) ** 2
    normalization = jnp.sum(timestep.valid) * ((self.config.transformations + 1)  ** 2)
    loss_v = jnp.sum(loss_v) / (normalization + (normalization == 0))
    
    
    return loss_v
  
  def abstraction_loss(self,
                       abstraction_params: Params,
                       iset_encoder_params: Params, 
                       similarity_params: Params, 
                       similarity_target: chex.Array,
                       public_state: chex.Array,
                       obs: chex.Array): 
    
    vectorized_abstraction = jax.vmap(self.abstraction_network.apply, in_axes=(None, 0), out_axes=0)
    vectorized_iset_encoder = jax.vmap(self.iset_encoder.apply, in_axes=(None, 0), out_axes=0) 
    vectorized_similarity = jax.vmap(jax.vmap(self.similarity_network.apply, in_axes=(None, 0), out_axes=0), in_axes=(None, -2), out_axes=-2)
    
    
    
    abstraction = vectorized_abstraction(abstraction_params, public_state)
    iset_probs = vectorized_iset_encoder(iset_encoder_params, obs)
    similarity = vectorized_similarity(similarity_params, abstraction) 
    
    # This computed the kmeans loss and the iset loss. TODO: Add the weighted term to the pi/v distance
    
    
    return _compute_soft_kmeans_loss_with_single(similarity_target, similarity, iset_probs) 
    
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
    iset_encoder_params: tuple[Params, Params],
    similarity_params: tuple[Params, Params],
    optimizers: Optimizers,
    similarity: chex.Array, # Expects to have 2nd to last dimension for playe
    timestep: TimeStep,
  ):
    if not self.config.train_abstraction:
      return abstraction_params, iset_encoder_params, similarity_params, optimizers
    abs_grad = []
    for pl in range(2):
      abstraction_loss, abstraction_grad = self._abstraction_loss(
        abstraction_params[pl], 
        iset_encoder_params[pl], 
        similarity_params[pl], 
        jax.lax.stop_gradient(similarity[..., pl, :]), 
        timestep.public_state, 
        timestep.obs[..., pl, :])
       
      
      abs_grad.append(abstraction_grad)
    
    abstraction_params = (*[optimizers.abstraction_optimizer[pl](abstraction_params[pl], abs_grad[pl][0]) for pl in range(2)],)
    iset_encoder_params = (*[optimizers.iset_encoder_optimizer[pl](iset_encoder_params[pl], abs_grad[pl][1]) for pl in range(2)],)
    similarity_params = (*[optimizers.similarity_optimizer[pl](similarity_params[pl], abs_grad[pl][2]) for pl in range(2)],)
    
    return abstraction_params, iset_encoder_params, similarity_params, optimizers
    
  def update_mvs_with_transformations(
    self,
    mvs_params: Params,
    mvs_params_target: Params,
    transformation_params: tuple[Params, Params],
    policy_params: Params,
    abstraction_params: tuple[Params, Params],
    iset_encoder_params: tuple[Params, Params],
    optimizers: Optimizers,
    pi_before_train: chex.Array,
    pi_after_train: chex.Array,
    timestep: TimeStep,
    
  ):
    if not self.config.train_mvs:
      return mvs_params, mvs_params_target, transformation_params, optimizers
    transform_grad = []
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
       
      transform_grad.append(transformation_grad)
     
    transformation_params = (*[optimizers.transformation_opitimizer[pl](transformation_params[pl], transform_grad[pl]) for pl in range(2)],)
    
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
    
    return mvs_params, mvs_params_target, transformation_params, optimizers
  
  def update_dynamics(
    self,
    dynamics_params: Params,
    abstraction_params: tuple[Params, Params],
    iset_encoder_params: tuple[Params, Params],
    optimizers: Optimizers,
    timestep: TimeStep
  ):
    if not self.config.train_dynamics:
      return dynamics_params, optimizers
    
    dynamics_loss, dynamics_grad = self._dynamics_loss(dynamics_params, abstraction_params, iset_encoder_params, timestep)
    
    dynamics_params = optimizers.dynamics_optimizer(dynamics_params, dynamics_grad)
    
    return dynamics_params, optimizers
  
  
  @functools.partial(jax.jit, static_argnums=(0,))
  def update_parameters(
    self,
    network_parameters: NetworkParameters,
    optimizers: Optimizers,
    timestep: TimeStep,
    alpha: float,
    update_net: bool, 
  ):
    
    expected_params, expected_params_target, optimizers = self.update_expected(
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
    
    
    rnad_params, rnad_params_target, rnad_params_prev, rnad_params_prev_, optimizers = self.update_rnad_with_expected(
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
    if self.config.similarity_metric == SimilarityMetric.POLICY_VALUE:
      similarity = jnp.concatenate(((pi * 2) - 1, v), axis=-1)
      # similarity = jnp.concatenate((pi, v), axis=-1)
    elif self.config.similarity_metric == SimilarityMetric.POLICY:
      similarity = (pi * 2) - 1
    elif self.config.similarity_metric == SimilarityMetric.VALUE:
      similarity = v
    elif self.config.similarity_metric == SimilarityMetric.LEGAL_ACTIONS:
      similarity = (timestep.legal * 2) - 1 # to be in range [-1, 1]
      
    
    
    abstraction_params, iset_encoder_params, similarity_params, optimizers = self.update_abstraction(
      network_parameters.abstraction_params,
      network_parameters.iset_encoder_params,
      network_parameters.similarity_params,
      optimizers,
      similarity,
      timestep
    )
    
    mvs_params, mvs_params_target, transformation_params, optimizers = self.update_mvs_with_transformations(
      network_parameters.mvs_params,
      network_parameters.mvs_params_target,
      network_parameters.transformation_params,
      rnad_params,
      abstraction_params,
      iset_encoder_params,
      optimizers,
      pi_before_train,
      pi,
      timestep
    )
    
    dynamics_params, optimizers = self.update_dynamics(
      network_parameters.dynamics_params,
      abstraction_params,
      iset_encoder_params,
      optimizers,
      timestep)
    
    
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
      iset_encoder_params=iset_encoder_params,
      similarity_params=similarity_params,
      dynamics_params=dynamics_params
      ), optimizers
  
  def step(self):
    trajectory = self.sample_trajectories()
    alpha, update_regularization = self._entropy_schedule(self.learner_steps)
    
    self.network_parameters, self.optimizers = self.update_parameters(
      self.network_parameters,
      self.optimizers,
      trajectory,
      alpha, 
      update_regularization)
     
    # self.params, self.params_target, self.params_prev, self.params_prev_, self.optimizer, self.optimizer_target = self.update_parameters(
    #   self.params, self.params_target, self.params_prev, self.params_prev_, self.optimizer, self.optimizer_target, trajectory, alpha, update_regularization)
    
    self.learner_steps += 1
    
  @functools.partial(jax.jit, static_argnums=(0,))
  def update_goofspiel_parameters(
    self,
    network_parameters: NetworkParameters,
    optimizers: Optimizers,
    key: chex.Array,
    alpha,
    update_net, 
  ):
    trajectory = self.sample_goofspiel_trajectories(network_parameters.rnad_params, key)
    
    
    return self.update_parameters(network_parameters, optimizers, lax.stop_gradient(trajectory), alpha, update_net)
  
  
  @functools.partial(jax.jit, static_argnums=(0,))
  def update_goofspiel_parameters_no_rnad(
    self,
    network_parameters: NetworkParameters,
    optimizers: Optimizers,
    key: chex.Array,
    alpha,
    update_net, 
  ):
    timestep = self.sample_goofspiel_trajectories(network_parameters.rnad_params, key)
     
    
    dynamics_params, optimizers = self.update_dynamics(
      network_parameters.dynamics_params,
      network_parameters.abstraction_params,
      network_parameters.iset_encoder_params,
      optimizers,
      timestep)
    
    
    return NetworkParameters(
      # rnad_params=rnad_params,
      # rnad_params_target=rnad_params_target,
      # rnad_params_prev=rnad_params_prev,
      # rnad_params_prev_=rnad_params_prev_,
      # # expected_params = expected_params,
      # # expected_params_target = expected_params_target,
      # mvs_params=mvs_params,
      # mvs_params_target=mvs_params_target,
      # transformation_params=transformation_params,
      # abstraction_params=abstraction_params,
      # iset_encoder_params=iset_encoder_params,
      # similarity_params=similarity_params,
      dynamics_params=dynamics_params
      ), optimizers
   
  def goofspiel_step(self):
    key = self.get_next_rng_key()
    alpha, update_regularization = self._entropy_schedule(self.learner_steps)
    
    self.network_parameters, self.optimizers = self.update_goofspiel_parameters(
      self.network_parameters,
      self.optimizers,
      key, 
      alpha, 
      update_regularization)
    
    self.learner_steps +=1
    
    
  def multiple_goofspiel_steps(self, iter: int):
    for _ in range(iter):
      self.goofspiel_step() 
      
  def multiple_goofspiel_steps2(self, iter: int):
    for _ in range(iter):
      key = self.get_next_rng_key()
      alpha, update_regularization = self._entropy_schedule(self.learner_steps)
      
      self.network_parameters, self.optimizers = self.update_goofspiel_parameters_no_rnad(
        self.network_parameters,
        self.optimizers,
        key, 
        alpha, 
        update_regularization)
      
      self.learner_steps +=1
      
      
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
      return rnad_params, rnad_params_target, rnad_params_prev, rnad_params_prev_, optimizers  
    rnad_loss, rnad_grad = self._rnad_with_expected_loss(rnad_params, rnad_params_prev, rnad_params_prev_,  expected_params_target, timestep, alpha)
    
    rnad_params = optimizers.rnad_optimizer(rnad_params, rnad_grad)
    
    rnad_params_target = optimizers.rnad_optimizer_target(
        rnad_params_target, jax.tree.map(lambda a, b: a - b, rnad_params_target, rnad_params))
    
    rnad_params_prev, rnad_params_prev_ = jax.lax.cond(
        update_net,
        lambda: (rnad_params_target, rnad_params_prev),
        lambda: (rnad_params_prev, rnad_params_prev_))
    return rnad_params, rnad_params_target, rnad_params_prev, rnad_params_prev_, optimizers  
    
  def v_trace_with_expected(
    self,
    state_v: chex.Array,
    valid: chex.Array,
    sampling_policy: chex.Array,
    network_policy: chex.Array,
    regularization_term: chex.Array,
    action_oh: chex.Array,  
    reward: chex.Array,
    lambda_: float = 1.0,
    c: float = 1.0,
    rho: float = 1.0,
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
    # rnad_params_target: Params,
    rnad_params_prev: Params,
    rnad_params_prev_: Params,
    # expected_params: Params,
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
    
    v_train_target, q_value = self.v_trace_with_expected(
      state_v, 
      expanded_valid, 
      timestep.policy, 
      pi, 
      regularized_term, 
      timestep.action, 
      timestep.reward, 
      c=self.config.c_iset_vtrace, 
      rho=self.config.rho_iset_vtrace, 
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
      return expected_params, expected_params_target, optimizers
    expected_loss, expected_grad = self._expected_loss(expected_params, expected_params_target, rnad_params, rnad_params_prev, rnad_params_prev_, timestep, alpha)
    
    expected_params = optimizers.expected_optimizer(expected_params, expected_grad)
    
    expected_params_target = optimizers.expected_optimizer_target(
        expected_params_target, jax.tree.map(lambda a, b: a - b, expected_params_target, expected_params))
    
    return expected_params, expected_params_target, optimizers
    
  def expected_v_trace(self,
                       v: chex.Array,
                       valid: chex.Array,
                       sampling_policy: chex.Array,
                       network_policy: chex.Array,
                       regularization_term: chex.Array,
                       action_oh: chex.Array,
                       reward: chex.Array,
                       lambda_: float = 1.0,
                       c: float = 1.0,
                       rho: float = 1.0,
                       eta: float = 0.2,
                       gamma: float = 1.0
                       ):
    importance_sampling = _policy_ratio(network_policy, sampling_policy, action_oh, valid[..., jnp.newaxis])
    regularization_entropy = eta * jnp.sum(network_policy * regularization_term, axis=-1)
    
    both_player_entrpy = regularization_entropy[..., 1] - regularization_entropy[..., 0]
    
    entropy_reward = jnp.expand_dims(reward + both_player_entrpy, -1)
    
    @chex.dataclass(frozen=True)
    class ExpectedVTraceCarry:
      next_value: chex.Array
      delta_v: chex.Array
    
    init_carry = ExpectedVTraceCarry(
      next_value=jnp.zeros_like(v[-1]),
      delta_v=jnp.zeros_like(v[-1])
    )
    
    def _expected_v_trace(carry: ExpectedVTraceCarry, x) -> tuple[ExpectedVTraceCarry, Any]:
      (importance_sampling, v, reward, valid) = x
      
      rho_ = jnp.prod(jnp.minimum(rho, importance_sampling), -2)
      c_ = jnp.prod(jnp.minimum(c, importance_sampling), -2)
      
      delta_v = rho_ * (reward + gamma * carry.next_value - v)
      carry_delta_v = delta_v + lambda_ * c_ * gamma * carry.delta_v
      
      v_target = v + carry_delta_v
      
      reset_carry = init_carry
      next_carry = ExpectedVTraceCarry(
        next_value=v,
        delta_v=carry_delta_v
      )
      return tree_where(valid, (next_carry, v_target), (reset_carry, jnp.zeros_like(v_target)))
    
    _, v_target = lax.scan(
      f=_expected_v_trace,
      init=init_carry,
      xs=(importance_sampling, v, entropy_reward, valid),
      reverse=True
    )
    return v_target 
    
    
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
    
    v_train_target = self.expected_v_trace(
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
  
  def extract_goofspiel_policy(self, game):
    assert isinstance(self.game, JaxOriginalGoofspiel)
    iset_set = []
    isets, legals = [], []
    def _traverse_tree(state: pyspiel.State, info):
      if state.is_terminal():
        return
      p1_iset = state.information_state_string(0)
      p2_iset = state.information_state_string(1)
      _, p1_goof_iset, p2_goof_iset, _ = self.game.get_info(info[0], info[1], info[2])
      if not p1_iset in iset_set:
        iset_set.append(p1_iset)
        isets.append(p1_goof_iset) 
        legals.append(info[3][0])
      if not p2_iset in iset_set:
        iset_set.append(p2_iset)
        isets.append(p2_goof_iset)
        legals.append(info[3][1])
      for a1 in state.legal_actions(0):
        for a2 in state.legal_actions(1):
          new_state = state.clone()
          new_state.apply_actions([a1, a2])
          new_legals, new_rewards, new_point_cards, new_played_cards, new_p1_points = self.game.apply_action(info[0], info[1], info[2], info[4], np.array([a1, a2]))
          _traverse_tree(new_state, (new_point_cards, new_played_cards, new_p1_points, new_legals, info[4]+1))
          
    init_info = self.game.initialize_structures()
    init_info = (*init_info, 0)
    _traverse_tree(game.new_initial_state(), init_info)
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
    for pl in range(2):
      self.optimizers.transformation_opitimizer[pl].state = state["optimizers"].transformation_opitimizer[pl]
      self.optimizers.abstraction_optimizer[pl].state = state["optimizers"].abstraction_optimizer[pl]
      self.optimizers.iset_encoder_optimizer[pl].state = state["optimizers"].iset_encoder_optimizer[pl]
      self.optimizers.similarity_optimizer[pl].state = state["optimizers"].similarity_optimizer[pl]
    
     


def compare_legals(muzero, game):

  init_info = game.initialize_structures()
  max_length = init_info[0].shape[0]
  
  ps, p1_isets, p2_isets = [], [], []
  p1_real_legals, p2_real_legals = [], []
  def _traverse_tree(info):
    if info[4] == max_length - 1:
      return
    _, p1_goof_iset, p2_goof_iset, public_state = game.get_info(info[0], info[1], info[2])
    ps.append(public_state)
    p1_isets.append(p1_goof_iset)
    p2_isets.append(p2_goof_iset)
    p1_real_legals.append(info[3][0])
    p2_real_legals.append(info[3][1])
    for a1_i, a1 in enumerate(info[3][0]):
      if a1 <= 0.5:
        continue
      for a2_i, a2 in enumerate(info[3][1]):
        if a2 <= 0.5:
          continue 
        new_legals, new_rewards, new_point_cards, new_played_cards, new_p1_points = game.apply_action(info[0], info[1], info[2], info[4], np.array([a1_i, a2_i]))
        _traverse_tree((new_point_cards, new_played_cards, new_p1_points, new_legals, info[4]+1))
  
  
  
  init_info = (*init_info, 0)
  _traverse_tree(init_info)
  ps = np.array(ps, dtype=np.float32)
  p1_isets = np.array(p1_isets, dtype=np.float32)
  p2_isets = np.array(p2_isets, dtype=np.float32)
  p1_real_legals = np.array(p1_real_legals, dtype=np.int8)
  p2_real_legals = np.array(p2_real_legals, dtype=np.int8)
  # print(ps.shape)
  # print(p1_isets.shape)
  # print(p2_isets.shape)
  # print(p1_real_legals.shape)
  # print(p2_real_legals.shape)
  p1_legals = muzero.get_legal_actions(ps, p1_isets, 0)
  p2_legals = muzero.get_legal_actions(ps, p2_isets, 1)
  
  # print(p1_legals.shape)
  # print(p2_legals.shape)
  p1_legals = p1_legals >= 0.0
  p2_legals = p2_legals >= 0.0
  
  p1_action_difference = np.sum(np.abs(p1_legals - p1_real_legals), -1)
  p2_action_difference = np.sum(np.abs(p2_legals - p2_real_legals), -1)
  p1_incorrect = np.sum(p1_action_difference > 0.1)
  p2_incorrect = np.sum(p2_action_difference > 0.1)
  print(p1_incorrect)
  print(p2_incorrect)
  
def goofspiel_compare_learned_trees(muzero, jax_game, orig_game):

  init_info = jax_game.initialize_structures()
  init_state = orig_game.new_initial_state()
  
  def _traverse_tree(info, state):
    if state.is_terminal():
      return
    _, p1_goof_iset, p2_goof_iset, public_state = jax_game.get_info(info[0], info[1], info[2])
    for a1 in state.legal_actions(0):
      for a2 in state.legal_actions(1):
        new_state = state.clone()
        new_state.apply_actions([a1, a2])
        new_legals, new_rewards, new_point_cards, new_played_cards, new_p1_points = jax_game.apply_action(info[0], info[1], info[2], info[4], np.array([a1, a2]))
        
        _, next_p1_goof_iset, next_p2_goof_iset, next_public_state = jax_game.get_info(new_point_cards, new_played_cards, new_p1_points)
        
        predicted_p1_iset, predicted_p2_iset, predicted_reward, predicted_terminal = muzero.get_next_state(public_state, p1_goof_iset, p2_goof_iset, a1, a2)
        
        # predicted_p1_iset, predicted_p2_iset, predicted_reward, predicted_terminal = muzero._jit_get_next_state(muzero.dynamics_params, p1_goof_iset, p2_goof_iset, a1, a2)
        
        abstracted_p1_iset, abstracted_p2_iset = muzero.get_both_abstraction(next_public_state, next_p1_goof_iset, next_p2_goof_iset)
        
        print(predicted_p1_iset)
        print(abstracted_p1_iset)
        print(jnp.linalg.norm(predicted_p1_iset - abstracted_p1_iset))
        print(predicted_p2_iset)
        print(abstracted_p2_iset)
        print(jnp.linalg.norm(predicted_p2_iset - abstracted_p2_iset))
        print(predicted_reward)
        print(new_rewards)
        print(new_state.is_terminal())
        print(predicted_terminal > 0.0)
        
        
        _traverse_tree((new_point_cards, new_played_cards, new_p1_points, new_legals, info[4]+1), new_state)
  
  
  init_info = (*init_info, 0)
  _traverse_tree(init_info, init_state)
  
  
from open_spiel.python.algorithms.best_response import BestResponsePolicy
from open_spiel.python.algorithms.mu_zero.jax_goofspiel import JaxOriginalGoofspiel

def main():
  cards = 5
  points_order = "descending"
  
  # params = {"num_cards": 5, "num_turns": 3, "first_round": 0}
  
  params = {"num_cards":cards, "points_order": points_order, "imp_info": True}
  
  orig_game =  pyspiel.load_game("goofspiel", params)
  
  game = JaxOriginalGoofspiel(cards, points_order)
  
  # game = orig_game 
  mu = True
  if mu == True:
    muzero = MuZero(game, MuZeroConfig(batch_size=32, trajectory_max=cards-1, use_abstraction=True, sampling_epsilon=0.0, entropy_schedule_size=(3000,)))
    # muzero.rng_key = jax.random.PRNGKey(42)
  else:
    params = [(n, p) for n, p in params.items()]
    muzero = RNaDSolver(RNaDConfig(game_name="goofspiel", game_params=params, trajectory_max=cards-1, batch_size=32))
  
  
  # profiler = Profiler()
  # profiler.start()
  with chex.fake_jit():
    for _ in range(60000):
      muzero.goofspiel_step()
     
  # goofspiel_compare_learned_trees(muzero, game, orig_game)
  # print("Ee")
  # compare_legals(muzero, game)
  # profiler.stop()
  # print(profiler.output_text(color=True, unicode=True))
  
  # policy = muzero.extract_full_policy()
     
  policy = muzero.extract_goofspiel_policy(orig_game)
  
  # print(policy.action_probabilities(state, 0))
  # print(policy.action_probabilities(state, 1))
  # print(state.information_state_tensor(0))
  # print(state.information_state_tensor(1))
  # # print(exploitability(game, policy))
  br1 = BestResponsePolicy(orig_game, 0, policy)
  br2 = BestResponsePolicy(orig_game, 1, policy)
  print(br1.value(orig_game.new_initial_state()))
  print(br2.value(orig_game.new_initial_state()))
  
  # traj = muzero.sample_trajectories()
  # print(traj.action.shape)
  
  
if __name__ == "__main__":
  main()