
import chex
import jax
import jax.numpy as jnp
from open_spiel.python.jax.cfr.jax_cfr import regret_matching

@chex.dataclass(frozen=True)
class MuZeroCFRConstants:
  """Constants for JaxCFR."""
 
  max_depth: int 
  max_actions: int

  max_iset_depth: chex.ArrayTree = ()  # Is just a list of integers
  depth_actions: chex.ArrayTree = ()  # Is just a list of integers
  depth_iset_map: chex.ArrayTree = () # ID -> Abstract iset for each D
  
  # Symbols: 
  #   D -> Depth
  #   Pl -> Amount of players
  #   H(D) -> Amount of histories at depth H(D)
  #   S(D) -> Amount of infosets at depth D
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

  iset_previous_action: chex.ArrayTree = () #Int[D, PL, S(D)]
  iset_action_mask: chex.ArrayTree = () #Int [D, PL, A]
  iset_action_depth: chex.ArrayTree = () 
  
  
class MuZeroCFR:
  def __init__(self, constants: MuZeroCFRConstants):
    self.constants = constants
    self.players = 2
    self._linear_averaging = True
    self._regret_matching_plus = True
    self._alternating_updates = True
    
    
    self.regrets = [[jnp.zeros((self.constants.depth_history_iset[d][pl].shape[0], a)) for pl in range(2)] for d, a in enumerate(constants.depth_actions)]
    print(self.regrets)
    self.averages = [[jnp.zeros((self.constants.depth_history_iset[d][pl].shape[0], a)) for pl in range(2)] for d, a in enumerate(constants.depth_actions)]

  def propagate_strategy(self, current_strategy):
    #Expecting current strategy to be [Depth, PL, S(D), A]
    reaches = jnp.ones_like(current_strategy[pl])
    for pl in range(self.players):
      for depth in range(1, self.constants.max_depth + 1):
        reaches[pl, depth, :, :] = reaches[pl, depth - 1, :, :] * current_strategy[pl, depth - 1, :, self.constants.iset_previous_action]
