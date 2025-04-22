import jax.numpy as jnp
import jax.lax as lax
import jax
import numpy as np
import chex
import optax
from typing import Any, Sequence
from enum import Enum



class SimilarityMetric(str, Enum):
  POLICY = "policy"
  VALUE = "value"
  POLICY_VALUE = "policy_value"
  LEGAL_ACTIONS = "legal_actions"
  LEGAL_POLICY_VALUE = "legal_policy_value"
  ACTION_HISTORY_POLICY = "action_history_policy"

class DynamicsType(str, Enum):
  ISET = "iset"
  PUBLIC_STATE = "public_state"

 
def _policy_ratio(pi: chex.Array, mu: chex.Array, actions_oh: chex.Array, valid: chex.Array) -> chex.Array: 
  pi_actions_prob = jnp.sum(pi * actions_oh, axis=-1, keepdims=True) * valid + (1 - valid)
  mu_actions_prob = jnp.sum(mu * actions_oh, axis=-1, keepdims=True) * valid + (1 - valid)
  
  return pi_actions_prob / mu_actions_prob

def tree_where(pred: chex.Array, x: chex.ArrayTree, y: chex.ArrayTree) -> chex.ArrayTree:
  """Apply jnp.where to each leaf of a pytree."""
  def _where(x, y):
    return jnp.where(pred, x, y)
  return jax.tree_map(_where, x, y)
  
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
  #norm = jnp.linalg.norm(x, 2, -1, keepdims=True)
  norm = jnp.sum(x ** 2, axis=-1, keepdims=True)
  norm = norm + (norm < 1e-15)
  norm = norm ** 0.5
  ret = x / norm
  return ret



def compute_soft_assignments(cluster_distance: chex.Array, temperature: float, cluster_closeness_assignment: float, repulsive_force: float):
  '''Computes a soft-assignments to soft k-means (fuzzy c-means). This is not the original method. When more than 1 point is too close to the center, we only move the closest one.'''
  closest = jnp.min(cluster_distance, -1, keepdims=True)
  # nulled_clusters = jnp.where(jnp.logical_and(cluster_distance < cluster_closeness_assignment, cluster_distance > closest + 1e-10), 0, 1)
  soft_assignments = jax.nn.softmax(-cluster_distance * temperature, axis=-1)
  
  soft_assignments = jnp.where(jnp.logical_and(cluster_distance < cluster_closeness_assignment, cluster_distance > closest + 1e-10), -soft_assignments * repulsive_force, soft_assignments)
  # soft_assignment = _legal_policy(-cluster_distance, nulled_clusters)
  return soft_assignments



def compute_energy_repulsion(pred: chex.Array):
  cluster_each_other_distance = pred[..., :, None, :] - pred[..., None, :, :]
  cluster_each_other_distance = jnp.sum(cluster_each_other_distance ** 2, axis=-1)
  
  exp_energy_repulsion = jnp.exp(-cluster_each_other_distance)
  
  cluster_energy_repulsion = jnp.where(cluster_each_other_distance < 1e-8, 0, exp_energy_repulsion)
  
  return jnp.mean(cluster_energy_repulsion)

def compute_energy_repulsion_inverse(pred: chex.Array):
  cluster_each_other_distance = pred[..., :, None, :] - pred[..., None, :, :]
  cluster_each_other_distance = jnp.sum(cluster_each_other_distance ** 2, axis=-1)
  
  cluster_energy_repulsion = jnp.where(cluster_each_other_distance < 1e-8, 0, 1/(cluster_each_other_distance + 1e-9))
  
  cluster_energy_repulsion = jnp.mean(cluster_energy_repulsion)
  return cluster_energy_repulsion


def _compute_soft_kmeans_sqrt_loss(real:chex.Array, pred: chex.Array, valid: chex.Array, temperature: float, cluster_closeness_assignment: float, repulsive_force: float):
  chex.assert_shape((real,), (*pred.shape[:-2], pred.shape[-1]))
  cluster_difference = lax.stop_gradient(jnp.expand_dims(real, -2)) - pred
  
  cluster_difference = cluster_difference * valid[..., None, None]
  
  cluster_distance = jnp.sum(cluster_difference ** 2, axis=-1)
  cluster_distance = cluster_distance + (cluster_distance < 1e-15)
  cluster_distance = cluster_distance ** 0.5
  
  cluster_soft_assignement = compute_soft_assignments(cluster_distance, temperature, cluster_closeness_assignment, repulsive_force)
    

    
  cluster_energy_repulsion = compute_energy_repulsion_inverse(pred) * 0.01
  
  
  cluster_loss = jnp.mean(cluster_difference ** 2, axis=-1)
  cluster_loss = jnp.sum(cluster_loss * cluster_soft_assignement, axis=-1) * valid
  return jnp.mean(cluster_loss) + cluster_energy_repulsion, cluster_soft_assignement
   

# TODO: This should take valid into account
def _compute_soft_kmeans_loss_with_cluster_assignments(real:chex.Array, pred: chex.Array, valid: chex.Array, temperature: float, cluster_closeness_assignment: float, repulsive_force: float):
  # The predicted dimension is missing
  chex.assert_shape((real,), (*pred.shape[:-2], pred.shape[-1]))
  cluster_difference = lax.stop_gradient(jnp.expand_dims(real, -2)) - pred
  
  cluster_difference = cluster_difference * valid[..., None, None]
  
  cluster_distance = jnp.sum(cluster_difference ** 2, axis=-1)
  # cluster_distance = cluster_distance + (cluster_distance < 1e-15)
  # cluster_distance = cluster_distance ** 0.5
  
  # cluster_loss = jax.nn.logsumexp(cluster_distance, axis=-1)
  
  cluster_soft_assignement = compute_soft_assignments(cluster_distance, temperature, cluster_closeness_assignment, repulsive_force)
    
  cluster_energy_repulsion = compute_energy_repulsion(pred) * 0.3
  
  
  cluster_loss = jnp.mean(cluster_difference ** 2, axis=-1)
  cluster_loss = jnp.sum(cluster_loss * cluster_soft_assignement, axis=-1) * valid
  return jnp.mean(cluster_loss) + cluster_energy_repulsion, cluster_soft_assignement
  
def _compute_soft_kmeans_loss_with_single(real: chex.Array, pred: chex.Array, probs: chex.Array, valid: chex.Array, temperature: float, cluster_closeness_assignment: float, repulsive_force: float):
  cluster_loss, cluster_soft_assignement = _compute_soft_kmeans_loss_with_cluster_assignments(real, pred, valid, temperature, cluster_closeness_assignment, repulsive_force)
  
  
  # cluster_soft_assignement = jnp.where(cluster_soft_assignement >= jnp.max(cluster_soft_assignement, -1, keepdims=True), 1, 0)
  # prob_loss = optax.losses.softmax_cross_entropy(probs,  jax.lax.stop_gradient(cluster_soft_assignement))
  
  
  # cluster_hard_assignement = jnp.argmax(cluster_soft_assignement, axis=-1)
  cluster_smooth_assignment = jnp.where(cluster_soft_assignement >= jnp.max(cluster_soft_assignement, -1, keepdims=True), 0.95, 0.05)
  
  prob_loss = optax.softmax_cross_entropy(probs, cluster_smooth_assignment)
  prob_loss = prob_loss * valid
  
  
  return cluster_loss + jnp.mean(prob_loss)




def v_trace( 
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



def retrace(
  q: chex.Array,
  valid: chex.Array,
  sampling_policy: chex.Array,
  network_policy: chex.Array,
  action_oh: chex.Array,
  reward: chex.Array, # Still not regularized
  lambda_: float = 1.0, # Lambda parameter for V-trace
  c: float = 1.0, # Importance sampling clipping
  rho: float = np.inf, # Importance sampling clipping
  gamma: float = 1.0 # Discount factor
):
  importance_sampling = _policy_ratio(network_policy, sampling_policy, action_oh, valid)
  
  # Clip importance sampling ratios
  importance_sampling = jnp.minimum(importance_sampling, rho)
  importance_sampling = jnp.minimum(importance_sampling, c)
    
  
  @chex.dataclass(frozen=True)
  class ReTraceCarry: 
    next_v: chex.Array # Policy * q in the next timestep
    delta_q: chex.Array # Propagated delta V in V-trace from the next timestep
  
  
  
  # Initialize carry for the scan
  init_carry = ReTraceCarry (
    next_v = jnp.zeros_like(q[-1]),  # Initialize with zeros for the last timestep
    delta_q = jnp.zeros_like(q[-1])  # Initialize delta Q with zeros
  )
  
  
  def _retrace(carry, x):
    importance_sampling_t, q_t, reward_t, valid_t, action_oh_t = x
    
    # TODO: We will compute next_v instead of next_q, since we have a policy in the next step, it is easy. We just need to replace the taken action with the target there.
    
    # This is for only the action taken. 
    delta_q = jnp.minimum(rho, importance_sampling_t) * (reward_t + gamma * carry.next_v - q_t)
      
    carry_delta_q = delta_q + lambda_ * jnp.minimum(c, importance_sampling_t) * gamma * carry.delta_q
    
    # Compute target Q
    q_target = q_t + carry_delta_q
    
    # Those 2 should be equivalent
    # next_q = action_oh_t * carry_delta_q + q_t
    next_q = action_oh_t * q_target + (1 - action_oh_t) * q_t
    next_v = jnp.sum(network_policy * next_q, axis=-1)
    
    # Update carry for next iteration
    next_carry = ReTraceCarry(
      next_v = next_v,
      delta_q = carry_delta_q
    )
    
    return next_carry, q_target
  
  # Run the scan in reverse order
  _, q_target = lax.scan(
    f=_retrace,
    init=init_carry,
    xs=(importance_sampling, q, reward, valid, action_oh),
    reverse=True
  )
  
  return q_target



def state_v_trace(
  
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



def expected_v_trace(
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
  
