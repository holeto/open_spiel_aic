
import chex
import jax
import jax.numpy as jnp

@chex.dataclass(frozen=True)
class MuZeroCFRConstants:
  """Constants for JaxCFR."""
 
  max_depth: int 
  max_actions: int

  max_iset_depth: chex.ArrayTree = ()  # Is just a list of integers
  depth_actions: chex.ArrayTree = ()  # Is just a list of integers
  depth_iset_map: chex.ArrayTree = () # ID -> Abstract iset
  
  # Symbols: 
  #   D -> Depth
  #   Pl -> Amount of players
  #   H(D) -> Amount of histories at depth H(D)
  #   A -> Actions of a player (has to be in junction with Pl)
  #   A1 -> Actions of P1
  #   A2 -> Actions of P2

  depth_history_action_utility: chex.ArrayTree = () # Float[D, H(D), A1, A2]
  depth_history_iset: chex.ArrayTree = () # Int[D, Pl, H(D)]
  depth_history_actions: chex.ArrayTree = () # Int[D, Pl, H(D), A] Just indices
  depth_history_legal: chex.ArrayTree = () # Bool[D, Pl, H(D), A] or [D, H(D), A1, A2]
  
  depth_history_previous_iset: chex.ArrayTree = () # Int[D, Pl, H(D)]
  depth_history_previous_action: chex.ArrayTree = () # Int[D, Pl, H(D)] (can be computed from previous_iset)
  depth_history_previous_history: chex.ArrayTree = () # Int[D, H(D)]

  depth_history_next_history: chex.ArrayTree = () # Int[D, H(D), A1, A2]

  iset_previous_action: chex.ArrayTree = ()
  iset_action_mask: chex.ArrayTree = ()
  iset_action_depth: chex.ArrayTree = ()
  
  
class MuZeroCFR:
  def __init__(self, constants):
    self.constants = constants
    self._linear_averaging = True
    self._regret_matching_plus = True
    self._alternating_updates = True
    
    
    self.regrets = [[jnp.zeros((self.constants.depth_history_iset[d][pl].shape[0], a)) for pl in range(2)] for d, a in enumerate(constants.depth_actions)]
    self.averages = [[jnp.zeros((self.constants.depth_history_iset[d][pl].shape[0], a)) for pl in range(2)] for d, a in enumerate(constants.depth_actions)]