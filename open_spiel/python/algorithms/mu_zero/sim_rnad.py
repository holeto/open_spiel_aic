
from open_spiel.python.algorithms.rnad.rnad import _legal_policy, legal_log_policy, EntropySchedule
from open_spiel.python.algorithms.mu_zero.flax_utils import init_network_with_optimizer, init_params_optimizer, optax_optimizer

from open_spiel.python.algorithms.rnad.rnad import RNaDSolver, RNaDConfig
from open_spiel.python.policy import TabularPolicy
from open_spiel.python.algorithms.exploitability import exploitability

from typing import Sequence, Any
from pyinstrument import Profiler
import jax
import jax.numpy as jnp
import jax.lax as lax

import flax.linen as nn
import chex
import optax

import numpy as np

import pyspiel

import functools

@chex.dataclass(frozen=True)
class TimeStep():
  
  valid: chex.Array = () # [..., 1]
  obs: chex.Array = () # [..., Player, O]
  legal: chex.Array = () # [..., Player, A]
  
  action: chex.Array = () # [..., Player, A]
  policy: chex.Array = () # [..., Player, A]
  
  reward: chex.Array = () # [..., 1] Reward after playing an action
  

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
    
    next_p1_iset = nn.Dense(self.isets)(x)
    next_p2_iset = nn.Dense(self.isets)(x)
    return next_p1_iset, next_p2_iset


# TODO: Do we want to send legal in? This forces us to train legal actions network
class RNaDNework(nn.Module):
  hidden_size: int
  out_dims: int
  
  
  @nn.compact
  def __call__(self, x, legal):
    x = nn.Dense(self.hidden_size)(x)
    x = nn.relu(x)
    x = nn.Dense(self.hidden_size)(x)
    x = nn.relu(x)
    logit = nn.Dense(self.out_dims)(x)
    v = nn.Dense(1)(x)
    
    pi = _legal_policy(logit, legal)
    log_pi = legal_log_policy(logit, legal)
    
    return pi, v, log_pi, logit
 
@chex.dataclass(frozen=True)
class MuZeroConfig: 
  
  batch_size: int = 32
  
  trajectory_max: int = 6
  
  abstraction_amount: int = 10
  abstraction_size: int = 32
  
  ps_hidden_size: int = 64
  iset_hidden_size: int = 64
  rnad_hidden_size: int = 256
  
  entropy_schedule_repeats: Sequence[int] = (1,)
  entropy_schedule_size: Sequence[int] = (2000,)
  
  learning_rate = 3e-4
  target_network_update = 1e-3
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
  advantage = q_values - jnp.sum(policy * logits, axis=-1, keepdims=True)
  advantage = advantage * importance_sampling
  advantage = lax.stop_gradient(jnp.clip(advantage, -clip, clip))
  mean_logit = jnp.sum(logits * legal, axis=-1, keepdims=True) / jnp.sum(legal, axis=-1, keepdims=True)
  
  logits_shifted = logits - mean_logit
  threshold_ceter = jnp.zeros_like(logits_shifted)
  
  neurd_loss_value = jnp.sum(legal * apply_force_with_threshold(logits_shifted, advantage, threshold, threshold_ceter), axis=-1, keepdims=True)
  
  return neurd_loss_value


# This contains RNaD implementation. Note that this implementation is specific for two-player zero-sum games. Unlike the open_spiel RNaD that can be used to general-sum multiplayer games.
class RNaDSimulataneous():
  def __init__(self, game, config) -> None:
    
    self.config = config
    self.game = game
    self.actions = game.num_distinct_actions()
    
    self.rng_key = jax.random.PRNGKey(self.config.seed)
    
    # temp_keys = self.get_next_rng_keys(6)
    
    self.example_state  = game.new_initial_state()
    self.example_timestep = self.default_timestep()
    
    self._entropy_schedule = EntropySchedule(
        sizes=self.config.entropy_schedule_size,
        repeats=self.config.entropy_schedule_repeats)
    
    self.rnad_network = RNaDNework(self.config.rnad_hidden_size, self.actions)
    
    self._rnad_loss = jax.value_and_grad(self.rnad_loss, has_aux=False)
    
    temp_key = self.get_next_rng_key()
    self.params = self.rnad_network.init(temp_key, self.example_timestep.obs, self.example_timestep.legal)
    
    # temp_keys = self.get_next_rng_keys(3)
    
    self.params_target = self.rnad_network.init(temp_key, self.example_timestep.obs, self.example_timestep.legal)
    self.params_prev = self.rnad_network.init(temp_key, self.example_timestep.obs, self.example_timestep.legal)
    self.params_prev_ = self.rnad_network.init(temp_key, self.example_timestep.obs, self.example_timestep.legal)
    
    self.optimizer = optax_optimizer(self.params, optax.chain(optax.adam(self.config.learning_rate, b1=0.0), optax.clip(100)))
    self.optimizer_target = optax_optimizer(self.params_target, optax.sgd(self.config.target_network_update))
    
    self.learner_steps = 0
  
    pass
  
  def default_timestep(self):
    obs = np.zeros(self.game.information_state_tensor_shape(), dtype=np.float32)
    
    legal = np.ones(self.actions, dtype=np.int8)
    action = np.ones(self.actions, dtype=np.float32)
    policy = np.ones(self.actions, dtype=np.float32)
    valid = np.array([0], dtype=np.float32)
    reward = np.array([0], dtype=np.float32)
    
    ts = TimeStep(
      valid = valid,
      obs = obs,
      legal = legal,
      action = action, 
      policy = policy,
      reward = reward
    )
    # return ts
    return jax.tree_util.tree_map(lambda xs: np.expand_dims(xs, 0), ts)
    

  @functools.partial(jax.jit, static_argnums=(0,))
  def _jit_get_network(self, params, obs, legal) -> chex.Array:
    return self.rnad_network.apply(params, obs, legal)
  
  @functools.partial(jax.jit, static_argnums=(0,))
  def _jit_get_policy(self, params, obs, legal) -> chex.Array:
    return self._jit_get_network(params, obs, legal)[0]
  
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
  
  # Expects obs and legal to be in shape [Batch, Player, ...]
  def batch_policy_and_action(self, obs, legal):
    
    keys = self.get_next_rng_keys_dimensional(obs.shape[:2])
    keys = np.array(keys)
    pi, action, action_oh = self._jit_get_batch_policy(self.params, keys, obs, legal)
    # pi, action, action_oh = self._jit_get_policy_and_action(self.params, keys, obs, legal)
    pi = np.array(pi, dtype=np.float64)
    pi = pi / np.sum(pi, axis=-1, keepdims=True) # TODO: Remove this
    action = np.array(action, dtype=np.int32)
    action_oh = np.array(action_oh, dtype=np.float64)
    return pi, action, action_oh
    
  def get_policy(self, state: pyspiel.State, player: int):
    obs = state.information_state_tensor(player) 
    legal = state.legal_actions_mask(player)
    pi = self._jit_get_policy(self.params, obs, legal)
    return np.array(pi, dtype=np.float32)
   
  def get_policy_both(self, state: pyspiel.State):
    obs = [state.information_state_tensor(pl) for pl in range(2)] 
    legal = [state.legal_actions_mask(pl) for pl in range(2)]
    obs = np.array(obs, dtype=np.float32)
    legal = np.array(legal, dtype=np.int8)
    pi = self._jit_get_policy(self.params, obs, legal)
    pi = np.array(pi, dtype=np.float64)
    return pi[0], pi[1]
  
  # TODO: Improve this
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
      _, p1_iset, p2_iset, _ = vectorized_get_info(carry.point_cards, carry.played_cards, carry.p1_points)
      obs = jnp.stack((p1_iset, p2_iset), axis=1)
      pi = network_apply(params, obs, carry.legal_actions)
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
    rho: float = 1.0, # Importance sampling clipping
    eta: float = 0.2, # Regularization factor 
    gamma: float = 1.0 # Discount factor
  ):
    
    importance_sampling = _policy_ratio(network_policy, sampling_policy, action_oh, valid)
    inverted_sampling = _policy_ratio(jnp.ones_like(sampling_policy), sampling_policy, action_oh, valid)
    
    regularization_entropy = eta * jnp.sum(network_policy * regularization_term, axis=-1)
    weighted_regularization_term = -eta * regularization_term + regularization_entropy[..., (1, 0), jnp.newaxis]
    
    both_player_entropy = (regularization_entropy[..., 1] - regularization_entropy[..., 0])
    
    entropy_reward = reward + both_player_entropy
    entropy_reward = jnp.expand_dims(jnp.stack((entropy_reward, -entropy_reward), axis=-1), -1)
    
    reward = jnp.expand_dims(jnp.stack((reward, -reward), axis=-1), -1)
    
    
    @chex.dataclass(frozen=True)
    class VTraceCarry: 
      next_value: chex.Array # Network value in the next timestep 
      delta_v: chex.Array # Propagated delta V in V-trace from the next timestep
    
    
    init_carry = VTraceCarry(
      next_value=jnp.zeros_like(v[-1]),
      delta_v=jnp.zeros_like(v[-1])
    )
  
    def _v_trace(carry: VTraceCarry, x) -> tuple[VTraceCarry, Any]:
      (importance_sampling, v, reward, entropy_reward, weighted_regularization_term, valid, inverted_sampling, action_oh) = x 
      # reward_uncorrected = reward + gamma * carry.reward_uncorrected + entropy
      # discounted_reward = reward + gamma * carry.reward
      
      delta_v = jnp.minimum(rho, importance_sampling) * (entropy_reward + gamma * carry.next_value - v)
      carry_delta_v = delta_v + lambda_ * jnp.minimum(c, importance_sampling) * gamma * carry.delta_v
      
      v_target = v + carry_delta_v
      
      # TODO: Shall we use opponent entropy reg term or entropy of played action?
      q_value = v + weighted_regularization_term  + action_oh * inverted_sampling  *(reward + gamma * importance_sampling * (carry.next_value + carry.delta_v) - v )
      
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
      xs=(importance_sampling, v, reward, entropy_reward, weighted_regularization_term, valid, inverted_sampling, action_oh),
      reverse=True
    ) 
    return v_target, q_value
  
  def rnad_loss(
    self,
    params: chex.ArrayTree,
    params_target: chex.ArrayTree,
    params_prev: chex.ArrayTree,
    params_prev_: chex.ArrayTree,
    timestep: TimeStep,
    alpha,
    learner_steps
  ):
    
    # We map over trajectory dimension and player dimension
    vectorized_net_apply = jax.vmap(jax.vmap(self.rnad_network.apply, in_axes=(None, 0, 0), out_axes=0), in_axes=(None, 1, 1), out_axes=1)
    
    pi, v, log_pi, logit = vectorized_net_apply(params, timestep.obs, timestep.legal)
    
    _, v_target, _, _ = vectorized_net_apply(params_target, timestep.obs, timestep.legal)
    _, _, log_pi_prev, _ = vectorized_net_apply(params_prev, timestep.obs, timestep.legal)
    _, _, log_pi_prev_, _ = vectorized_net_apply(params_prev_, timestep.obs, timestep.legal)
    
    # This creates the regularization term for rewards
    regularized_term = log_pi - (alpha * log_pi_prev + (1 - alpha) * log_pi_prev_) 
    
    expanded_valid = jnp.expand_dims(timestep.valid, (-2, -1))
    
    v_train_target, q_value = self.v_trace(v_target, expanded_valid, timestep.policy, pi, regularized_term, timestep.action, timestep.reward)
    
    # We multiply by 2, since each player acts
    normalization = jnp.sum(timestep.valid) * 2 
    v_loss = jnp.sum((expanded_valid * (v - lax.stop_gradient(v_train_target)) ** 2)) / (normalization + (normalization == 0))
    
    importance_sampling = jnp.where(timestep.legal, pi / timestep.policy, 0)
    
    loss_neurd = neurd_loss(logit, pi, q_value, timestep.legal, importance_sampling)
    
    neurd_loss_value = -jnp.sum(loss_neurd * expanded_valid) / (normalization + (normalization == 0))
    return v_loss + neurd_loss_value
  
  
  @functools.partial(jax.jit, static_argnums=(0,))
  def update_parameters(
    self,
    params: chex.ArrayTree,
    params_target: chex.ArrayTree,
    params_prev: chex.ArrayTree,
    params_prev_: chex.ArrayTree,
    optimizer,
    optimizer_target,
    timestep: TimeStep,
    alpha,
    learner_steps,
    update_net, 
  ):
    
    loss, grad = self._rnad_loss(params, params_target, params_prev, params_prev_, timestep, alpha, learner_steps)
    
    params = optimizer(params, grad)
    
    params_target = optimizer_target(
        params_target, jax.tree.map(lambda a, b: a - b, params_target, params))
    
    params_prev, params_prev_ = jax.lax.cond(
        update_net,
        lambda: (params_target, params_prev),
        lambda: (params_prev, params_prev_))
    return params, params_target, params_prev, params_prev_, optimizer, optimizer_target
  
  def step(self):
    trajectory = self.sample_trajectories()
    alpha, update_regularization = self._entropy_schedule(self.learner_steps)
    
    self.params, self.params_target, self.params_prev, self.params_prev_, self.optimizer, self.optimizer_target = self.update_parameters(
      self.params, self.params_target, self.params_prev, self.params_prev_, self.optimizer, self.optimizer_target, trajectory, alpha, self.learner_steps, update_regularization)
    
    self.learner_steps += 1
    
  @functools.partial(jax.jit, static_argnums=(0,))
  def update_goofspiel_parameters(
    self,
    params: chex.ArrayTree,
    params_target: chex.ArrayTree,
    params_prev: chex.ArrayTree,
    params_prev_: chex.ArrayTree,
    optimizer,
    optimizer_target,
    key: chex.Array,
    alpha,
    learner_steps,
    update_net, 
  ):
    trajectory = self.sample_goofspiel_trajectories(self.params, key)
    self.params, self.params_target, self.params_prev, self.params_prev_, self.optimizer, self.optimizer_target = self.update_parameters(
      self.params, self.params_target, self.params_prev, self.params_prev_, self.optimizer, self.optimizer_target, trajectory, alpha, self.learner_steps, update_net)
    
    return self.params, self.params_target, self.params_prev, self.params_prev_, self.optimizer, self.optimizer_target
    
  def goofspiel_step(self):
    key = self.get_next_rng_key()
    alpha, update_regularization = self._entropy_schedule(self.learner_steps)
    self.params, self.params_target, self.params_prev, self.params_prev_, self.optimizer, self.optimizer_target = self.update_goofspiel_parameters(
      self.params, self.params_target, self.params_prev, self.params_prev_, self.optimizer, self.optimizer_target, key, alpha, self.learner_steps, update_regularization)
    
    self.learner_steps +=1
    
    
    
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
    pi = self._jit_get_policy(self.params_target, isets, legals)
    policy = TabularPolicy(self.game)
    # policy.
    for i, iset in enumerate(iset_set):
      policy.action_probability_array[policy.state_lookup[iset]] = pi[i]
  
    return policy
  
  
  
  

  
  
from open_spiel.python.algorithms.best_response import BestResponsePolicy
from open_spiel.python.algorithms.mu_zero.jax_goofspiel import JaxOriginalGoofspiel

def main():
  cards = 5
  points_order = "descending"
  
  # params = {"num_cards": 5, "num_turns": 3, "first_round": 0}
  
  params = {"num_cards":cards, "points_order": points_order, "imp_info": True}
  
  game =  pyspiel.load_game("goofspiel", params)
  
  # game = JaxOriginalGoofspiel(cards, points_order)
   
  mu = True
  if mu == True:
    muzero = RNaDSimulataneous(game, MuZeroConfig(batch_size=32, trajectory_max=4))
    # muzero.rng_key = jax.random.PRNGKey(42)
  else:
    params = [(n, p) for n, p in params.items()]
    muzero = RNaDSolver(RNaDConfig(game_name="goofspiel", game_params=params, trajectory_max=4, batch_size=32))
    muzero._rngkey = jax.random.PRNGKey(42)
  
  
  profiler = Profiler()
  profiler.start()
  for _ in range(10000):
    muzero.step()
    
  # policy = muzero.extract_full_policy()
  
  # state = game.new_initial_state()
  # state.apply_actions([0, 1])
  # print(policy.action_probabilities(state, 0))
  # print(policy.action_probabilities(state, 1))
  # print(state.information_state_tensor(0))
  # print(state.information_state_tensor(1))
  # # print(exploitability(game, policy))
  # br1 = BestResponsePolicy(game, 0, policy)
  # br2 = BestResponsePolicy(game, 1, policy)
  # print(br1.value(game.new_initial_state()))
  # print(br2.value(game.new_initial_state()))
  
    
  profiler.stop()
  print(profiler.output_text(color=True, unicode=True))
  # traj = muzero.sample_trajectories()
  # print(traj.action.shape)
  
  
if __name__ == "__main__":
  main()