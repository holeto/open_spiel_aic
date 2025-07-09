import numpy as np
import pickle
import chex
import jax.numpy as jnp
import jax

from typing import Any



def tree_where(pred: chex.Array, x: chex.ArrayTree, y: chex.ArrayTree) -> chex.ArrayTree:
  """Apply jnp.where to each leaf of a pytree."""
  def _where(x, y):
    return jnp.where(pred, x, y)
  return jax.tree_map(_where, x, y)


#Not needed for now
# if we later want to use value as train target
def state_v_trace(

v: chex.Array, # Current value estimates, in our case output of afterstate prediction
sampling_policy: chex.Array, # The reference sampling policy
target_policy: chex.Array, # Target policy, the actual reference policy we want to learn (in this case same as sampling, so essentially just getting n-step Bellman targets)
actions_oh: chex.Array,
valid: chex.Array, # Will always be 1 here
reward: chex.Array,
lambda_: float = 1.0, # Lambda parameter for V-trace
c: float = 1.0, # Importance sampling clipping
rho: float = np.inf, # Importance sampling clipping.
gamma: float = 1.0 # Discount factor
) -> chex.Array:
  pi_action_prob = jnp.sum(target_policy * actions_oh, axis=-1)
  mu_action_prob = jnp.sum(sampling_policy * actions_oh, axis=-1)
  importance_sampling = pi_action_prob / mu_action_prob
  
  #TODO: Check the shapes, this is probably not correct
  # Policies should be of shape [Trajectory, Batch, Player, actions]
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
  
  _, v_target = jax.lax.scan(
    f=_state_v_trace,
    init=init_carry,
    xs=(p1_is, p2_is, v, jnp.expand_dims(reward, (-1, -2)), jnp.expand_dims(valid, (-1, -2))),
    reverse=True
  )

  return v_target

def get_reference_policy(game_state, legal_actions):
  #TODO: The fixed reference policy goes here, for now uniform
  #return legal_actions / jnp.sum(legal_actions, axis=-1, keepdims=True)
  #Play the highest legal action available
  max_actions = jnp.argmax(jnp.cumsum(legal_actions, axis=-1), axis=-1)
  actions_oh = jax.nn.one_hot(max_actions, num_classes=legal_actions.shape[1], axis=-1)
  return actions_oh


def check_param_difference(p_after, p_before):
  """Debug method to check the change for parameters
  after an update"""
  diff_tree = jax.tree_util.tree_map(lambda p_after, p_before: p_after - p_before, p_after, p_before)
  norm_tree = jax.tree_util.tree_map(lambda x: jnp.linalg.norm(x), diff_tree)
  print(norm_tree)

  
def pickle_dump(filename, data):
  with open(filename, "wb") as f:
    pickle.dump(data, f)

def pickle_load(filename):
  with open(filename, "rb") as f:
    data = pickle.load(f)
  return data