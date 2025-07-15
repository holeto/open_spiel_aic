"""Contains the afterstate, encoder and dynamics networks
of VQ-VAE as used in Stochastic Muzero. For now, the architecture
of the used networks is exactly the same as in the original implementation
https://github.com/DHDev0/Stochastic-muzero"""

import jax
import jax.numpy as jnp
from flax import nnx


class Afterstate_decoder_function(nnx.Module):
  """ Models a function that returns the original
  state corresponding to the given afterstate."""
  def __init__(self,
                afterstate_dimension,
                state_dimension,
                hidden_layer_dimension,
                rngs: nnx.Rngs):
      
      self.linear_in = nnx.Linear(afterstate_dimension, hidden_layer_dimension, rngs=rngs)
      self.linear_mid = nnx.Linear(hidden_layer_dimension, hidden_layer_dimension, rngs=rngs)
      #policy which we will want to match on the actual exploring policy
      self.linear_out_state = nnx.Linear(hidden_layer_dimension, state_dimension, rngs=rngs)

      self.activation = nnx.elu

  #TODO: Check whether the afterstate
  # is transformed in some way before being passed to the network
  def __call__(self, afterstate):
    x = self.activation(self.linear_in(afterstate))
    x = self.activation(self.linear_mid(x))
    orig_state = self.linear_out_state(x)
    return orig_state

class Afterstate_representation_function(nnx.Module):
  """Receive an actual state.
     Return the afterstate  
    """
  def __init__(self,
                state_dimension,
                afterstate_dimension,
                hidden_layer_dimension,
                rngs: nnx.Rngs):
      
      self.linear_in = nnx.Linear(state_dimension, hidden_layer_dimension, rngs=rngs)
      self.linear_mid = nnx.Linear(hidden_layer_dimension, hidden_layer_dimension, rngs=rngs)
      #Afterstate induced by state
      self.linear_out_afterstate= nnx.Linear(hidden_layer_dimension,afterstate_dimension, rngs=rngs)

      self.activation = nnx.elu

  def __call__(self, state):
   x = self.activation(self.linear_in(state))
   x = self.activation(self.linear_mid(x))
   afterstate = self.linear_out_afterstate(x)
   return afterstate
  
class Codebook_function(nnx.Module):
  """Receive an aftestate and return 
  action_dimension next possible afterstates.
  These are returned as action_dimension x afterstate_dimension matrix"""

  def __init__(self,
                action_dimension,
                afterstate_dimension,
                hidden_layer_dimension,
                rngs: nnx.Rngs):
      
      self.linear_in = nnx.Linear(afterstate_dimension, hidden_layer_dimension, rngs=rngs)
      self.linear_mid = nnx.Linear(hidden_layer_dimension, hidden_layer_dimension, rngs=rngs)
      #Afterstate induced by state
      self.linear_out_codebook= nnx.Linear(hidden_layer_dimension,action_dimension * afterstate_dimension, rngs=rngs)

      self.activation = nnx.elu
      self.action_dimension = action_dimension
      self.afterstate_dimension = afterstate_dimension

  def __call__(self, afterstate):
   x = self.activation(self.linear_in(afterstate))
   x = self.activation(self.linear_mid(x))
   flat_codebook = self.linear_out_codebook(x)
   codebook = jnp.reshape(flat_codebook, (*flat_codebook.shape[:-1], self.action_dimension, self.afterstate_dimension))
   return codebook


class Policy_function(nnx.Module):
  """Receive next state tensor,
    Return: softmaxed logits c_e of 
    the same size as action space corresponding
    to  distribution over the next_outcomes (in our simple case, the next aftestate),
    c corresponding to the category of the next outcome as 
    one_hot(argmax(c_e))."""
  def __init__(self,
                afterstate_dimension,
                action_dimension,
                hidden_layer_dimension,
                rngs: nnx.Rngs):
      self.linear_in = nnx.Linear(afterstate_dimension, hidden_layer_dimension, rngs=rngs)
      self.linear_mid = nnx.Linear(hidden_layer_dimension, hidden_layer_dimension, rngs=rngs)
      self.linear_out_policy= nnx.Linear(hidden_layer_dimension,action_dimension, rngs=rngs)

      self.activation = nnx.elu
      self.action_dimension = action_dimension

  def __call__(self, afterstate):
     x = self.activation(self.linear_in(afterstate))
     x = self.activation(self.linear_mid(x))
     policy_logits = self.linear_out_policy(x)
     #TODO: Verify if this is correct, we want a straight through estimator
     #c = nnx.one_hot(jnp.argmax(policy_logits, axis=-1),self.action_dimension, axis=-1) - jax.lax.stop_gradient(policy_logits) + policy_logits
     return policy_logits
  

class Afterstate_dynamics_function(nnx.Module):
  """
  Receive last afterstate and the 
  one hot encoded action a
  and return next afterstate
  """
  def __init__(self,
              afterstate_dimension,
              action_dimension,
              hidden_layer_dimension,
              rngs: nnx.Rngs):
    
    self.linear_in = nnx.Linear(afterstate_dimension + action_dimension, hidden_layer_dimension, rngs=rngs)
    self.linear_mid = nnx.Linear(hidden_layer_dimension, hidden_layer_dimension, rngs=rngs)
    self.linear_out_next_afterstate= nnx.Linear(hidden_layer_dimension,afterstate_dimension, rngs=rngs)

    self.activation = nnx.elu

  def __call__(self, state, c):
   x = jnp.concatenate([jnp.ravel(state),jnp.ravel(c)])
   x = self.activation(self.linear_in(x))
   x = self.activation(self.linear_mid(x))
   next_afterstate = self.linear_out_next_afterstate(x)
   return next_afterstate