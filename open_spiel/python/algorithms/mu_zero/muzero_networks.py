import jax.numpy as jnp
import chex
from flax import linen as nn
from typing import Any, Sequence
from open_spiel.python.algorithms.rnad.rnad import _legal_policy, legal_log_policy
  
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

class PublicStateDynamicsNetwork(nn.Module):
  hidden_size: int
  public_state_size: int
  abstraction_amount: int
  
  @nn.compact
  def __call__(self, p1_isets, p2_isets, p1_action, p2_action):
    x = jnp.concatenate((p1_isets, p2_isets, p1_action, p2_action), axis=-1)
    x = nn.Dense(self.hidden_size)(x)
    x = nn.relu(x)
    x = nn.Dense(self.hidden_size)(x)
    x = nn.relu(x)
    public_state = nn.Dense(self.public_state_size)(x)
    p1_dist = nn.Dense(self.abstraction_amount)(x)
    p2_dist = nn.Dense(self.abstraction_amount)(x)
    reward = nn.Dense(2)(x)
    is_terminal = nn.Dense(1)(x)
    return public_state, p1_dist, p2_dist, reward, is_terminal

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
    ps = ps.reshape(*ps.shape[:-1], self.isets, self.iset_size)
    return ps

class PublicStateDecoder(nn.Module):
  hidden_size: int
  public_state_size: int
  
  @nn.compact
  def __call__(self, x: chex.Array) -> chex.Array:
    x = nn.Dense(self.hidden_size)(x)
    x = nn.relu(x)
    x = nn.Dense(self.hidden_size)(x)
    x = nn.relu(x)
    ps = nn.Dense(self.public_state_size)(x)
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
    x = x.reshape(*x.shape[:-1], self.transformations, self.actions)
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
    x = x.reshape(*x.shape[:-1], self.values, self.values)
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
    x = x.reshape(*x.shape[:-1], 2, self.values)
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
 
class RNaDNetwork(nn.Module):
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

class QCriticNetwork(nn.Module):
  hidden_size: int
  actions: int

  @nn.compact
  def __call__(self, x: chex.Array) -> chex.Array:
    x = nn.Dense(self.hidden_size)(x)
    x = nn.relu(x)
    x = nn.Dense(self.hidden_size)(x)
    x = nn.relu(x)
    q = nn.Dense(self.actions)(x)
    return q