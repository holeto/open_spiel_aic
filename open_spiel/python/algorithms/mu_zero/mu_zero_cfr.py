
import chex
import jax
import jax.numpy as jnp
import functools
import numpy as np

from open_spiel.python.jax.cfr.jax_cfr import JAX_CFR_SIMULTANEOUS_UPDATE, regret_matching


def check_iset_similarity(iset1, iset2, threshold=0.05):
  # return jnp.mean(jnp.abs(iset1 - iset2)) < threshold
  similarity = np.linalg.norm(iset1 - iset2, ord=2) / np.sqrt(iset1.shape[-1])
  return similarity < threshold


@chex.dataclass(frozen=True)
class MuZeroCFRConstants:
  """Constants for JaxCFR."""
 
  max_depth: int 
  resolving_player: int

  init_reaches: chex.Array = ()

  depth_actions: chex.ArrayTree = ()  # Is just a list of integers
  
  depth_iset_map: chex.ArrayTree = () # ID -> Abstract iset
  depth_iset_legal: chex.ArrayTree = ()
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

  depth_history_next_history: chex.ArrayTree = () # Int[D, H(D), A1, A2]
 
  
  
class MuZeroCFR:
  def __init__(self, constants: MuZeroCFRConstants):
    
    self.check_constants(constants)
    
    self.constants = constants
    self.players = 2
    self._linear_averaging = True
    self._regret_matching_plus = True
    self._alternating_updates = True
    
    self.timestep = 1
    
    self.regrets = [[jnp.zeros((self.constants.depth_iset_legal[d][pl].shape[0], a)) for pl in range(self.players)] for d, a in enumerate(constants.depth_actions)]
    self.averages = [[jnp.zeros((self.constants.depth_iset_legal[d][pl].shape[0], a)) for pl in range(self.players)] for d, a in enumerate(constants.depth_actions)]
    self.cf_values = [[jnp.zeros((self.constants.depth_iset_legal[d][pl].shape[0]),) for pl in range(self.players)] for d, _ in enumerate(constants.depth_actions)]
    
    self.regret_matching = jax.vmap(regret_matching, in_axes=(0, 0), out_axes=0)
    
  def check_constants(self, constants: MuZeroCFRConstants):
    # TODO: Change those to chex
    assert constants.max_depth > 0, "Max depth has to be at least 1"
    assert constants.resolving_player in [0, 1], "Resolving player has to be 0 or 1"
    assert constants.init_reaches.shape[0] == 2, "Reaches have to be for both players"
    assert len(constants.depth_actions) == constants.max_depth, "Depth actions have to be defined for each depth"
    assert len(constants.depth_iset_map) == constants.max_depth, "Depth iset map has to be defined for each depth"
    assert len(constants.depth_iset_legal) == constants.max_depth, "Depth iset legal has to be defined for each depth"
    assert len(constants.depth_history_action_utility) == constants.max_depth, "Depth history action utility has to be defined for each depth"
    assert len(constants.depth_history_iset) == constants.max_depth, "Depth history iset has to be defined for each depth"
    assert len(constants.depth_history_actions) == constants.max_depth, "Depth history actions has to be defined for each depth"
    assert len(constants.depth_history_legal) == constants.max_depth, "Depth history legal has to be defined for each depth"
    assert len(constants.depth_history_next_history) == constants.max_depth, "Depth history next history has to be defined for each depth" 
    for d in range(constants.max_depth):
      histories_in_depth = constants.depth_history_iset[d].shape[1]
      for pl in range(2):
        assert constants.depth_iset_legal[d][pl].shape[1] == constants.depth_actions[d]
        # assert constants.dept
      assert constants.depth_history_action_utility[d].shape[0] == histories_in_depth
      assert constants.depth_history_action_utility[d].shape[1] == constants.depth_actions[d]
      assert constants.depth_history_action_utility[d].shape[2] == constants.depth_actions[d]
      assert constants.depth_history_next_history[d].shape[0] == histories_in_depth
      assert constants.depth_history_next_history[d].shape[1] == constants.depth_actions[d]
      assert constants.depth_history_next_history[d].shape[2] == constants.depth_actions[d]
      assert constants.depth_history_legal[d].shape[0] == histories_in_depth
      assert constants.depth_history_legal[d].shape[1] == constants.depth_actions[d]
      assert constants.depth_history_legal[d].shape[2] == constants.depth_actions[d] 
      assert constants.depth_history_actions[d].shape[1] == histories_in_depth
    
  def multiple_steps(self, iterations: int):
    for _ in range(iterations):
      self.step()
    
  def step(self):
    """Wrapper around the jitted function for performing CFR step."""
    averaging_coefficient = self.timestep if self._linear_averaging else 1
    if self._alternating_updates:
      for player in range(self.players):
        self.regrets, self.averages, self.cf_values = self.jit_step(
            self.regrets, self.averages, self.cf_values, averaging_coefficient, player, self.timestep
        )

    else:
      self.regrets, self.averages, self.cf_values = self.jit_step(
          self.regrets,
          self.averages,
          self.cf_values,
          averaging_coefficient,
          JAX_CFR_SIMULTANEOUS_UPDATE, # TODO: Use constant from JaxCFR
          self.timestep
      )

    self.timestep += 1
  

  def get_strategy(self, iset, player, depth):
    iset_id = self.find_most_likely_index(iset, player, depth)
    
    iset_strategy = self.averages[depth][player][iset_id]
    normalization = jnp.sum(iset_strategy)
    
    return jnp.where(normalization > 1e-10, iset_strategy / normalization, 1 / self.averages[depth][player].shape[-1])
  
  def find_public_state_from_iset(self, iset, player, depth):
    init_infoset = self.find_most_likely_index(iset, player, depth)
    visited_isets = [set(), set()]
    visited_histories = set()
    curr_player = player
    # TODO: Use sets?
    curr_isets = np.array([init_infoset])
    visited_isets[player].add(init_infoset)
    while curr_isets.size > 0:
      matching_iset = self.constants.depth_history_iset[depth][curr_player, ..., None] == curr_isets[None, ...]
      histories = np.nonzero(np.sum(matching_iset, -1))
      curr_player = 1 -curr_player
      next_isets = []
      for history in histories[0]:
        visited_histories.add(int(history))
        history_iset = int(self.constants.depth_history_iset[depth][curr_player][history])
        if history_iset not in visited_isets[curr_player]:
          next_isets.append(history_iset)
          visited_isets[curr_player].add(history_iset)
      curr_isets = np.array(next_isets)
      
    return np.fromiter(visited_histories, int)#, np.fromiter(visited_isets[0], int), np.fromiter(visited_isets[1], int)
    
  
  def find_most_likely_index(self, iset, player, depth):
    closeness = np.linalg.norm(iset - self.constants.depth_iset_map[depth][player], axis=-1)
    return np.argmin(closeness)
  
  def find_iset_index(self, iset, player, depth):
    for i in range(self.constants.depth_iset_map[depth][player].shape[0]):
      if check_iset_similarity(iset, self.constants.depth_iset_map[depth][player][i]):
        return i
    return -1

  def get_last_depth_player_cf_values(self, player):
    return self.cf_values[-1][player][self.constants.depth_history_iset[-1][player]]
    
  def find_reaches_from_average(self):
    averages = [[self.averages[d][pl] / jnp.sum(self.averages[d][pl], axis=-1, keepdims=True) for pl in range(self.players)] for d in range(self.constants.max_depth)]
    return self.find_reaches(averages)
    
  # TODO: Could this be used in jit_step?
  def find_reaches(self, strategies):
  
    history_reaches = [self.constants.init_reaches]
    history_strategies = [jnp.stack([strategies[d][pl][self.constants.depth_history_iset[d][pl]] for pl in range(self.players)], axis=0) for d in range(self.constants.max_depth)]
    
    
    for d in range(self.constants.max_depth - 1):
      strategy_realization = history_reaches[d][..., None] * history_strategies[d]

      
      p1_masked_realization = strategy_realization[0, ..., None] * (self.constants.depth_history_next_history[d] >= 0)
      
      p2_masked_realization = strategy_realization[1, :, None, ...] * (self.constants.depth_history_next_history[d] >= 0)
      
      
      p1_reaches_next = jnp.bincount(self.constants.depth_history_next_history[d].ravel(), p1_masked_realization.ravel(), length=self.constants.depth_history_next_history[d+1].shape[0])
      p2_reaches_next = jnp.bincount(self.constants.depth_history_next_history[d].ravel(), p2_masked_realization.ravel(), length=self.constants.depth_history_next_history[d+1].shape[0])
      
      history_reaches.append(jnp.stack([p1_reaches_next, p2_reaches_next], axis=0))
    return history_reaches
    
    
  # Is it okay to compile for each player separately?
  # TODO: Could we remove some bin counts?
  @functools.partial(jax.jit, static_argnums=(0, 5))
  def jit_step(self, regrets, averages, cf_values, average_policy_update_coefficient, player, iteration):
    
    current_strategies = [[self.regret_matching(regrets[d][pl], self.constants.depth_iset_legal[d][pl]) for pl in range(self.players)] for d in range(self.constants.max_depth)]
  
    history_reaches = [self.constants.init_reaches]
    history_strategies = [jnp.stack([current_strategies[d][pl][self.constants.depth_history_iset[d][pl]] for pl in range(self.players)], axis=0) for d in range(self.constants.max_depth)]
    
    
    for d in range(self.constants.max_depth):
      strategy_realization = history_reaches[d][..., None] * history_strategies[d]
      
      # TODO: This is dumb 
      if player != 1:
        p1_iset_realizations = jnp.bincount(self.constants.depth_history_actions[d][0].ravel(), strategy_realization[0].ravel(), length=self.constants.depth_actions[d] * self.constants.depth_iset_legal[d][0].shape[0]).reshape(averages[d][0].shape)
        averages[d][0] = averages[d][0] + p1_iset_realizations * average_policy_update_coefficient
        
      if player != 0:
        p2_iset_realizations = jnp.bincount(self.constants.depth_history_actions[d][1].ravel(), strategy_realization[1].ravel(), length=self.constants.depth_actions[d] * self.constants.depth_iset_legal[d][1].shape[0]).reshape(averages[d][1].shape) 
        averages[d][1] = averages[d][1] + p2_iset_realizations * average_policy_update_coefficient
      
      # We do not compute the next history, since there is none
      if d == self.constants.max_depth - 1:
        break
      
      p1_masked_realization = strategy_realization[0, ..., None] * (self.constants.depth_history_next_history[d] >= 0)
      
      p2_masked_realization = strategy_realization[1, :, None, ...] * (self.constants.depth_history_next_history[d] >= 0)
      
      
      p1_reaches_next = jnp.bincount(self.constants.depth_history_next_history[d].ravel(), p1_masked_realization.ravel(), length=self.constants.depth_history_next_history[d+1].shape[0])
      p2_reaches_next = jnp.bincount(self.constants.depth_history_next_history[d].ravel(), p2_masked_realization.ravel(), length=self.constants.depth_history_next_history[d+1].shape[0])
      
      history_reaches.append(jnp.stack([p1_reaches_next, p2_reaches_next], axis=0))
      
    # How to work with depth_utils in the first round (and subsequent)
    depth_utils = [jnp.zeros((1,))]
    
    for d in range(self.constants.max_depth - 1, -1, -1):
      action_value = jnp.where(self.constants.depth_history_next_history[d] >= 0, depth_utils[-1][self.constants.depth_history_next_history[d]], self.constants.depth_history_action_utility[d])
      # action_value = self.constants.depth_history_action_utility[d] + depth_utils[-1][self.constants.depth_history_next_history[d]] * (self.constants.depth_history_next_history[d] >= 0)
      action_probabilities = history_strategies[d][0,..., None] * history_strategies[d][1, :, None, ...]
      history_value = jnp.sum(action_value * action_probabilities, axis=(-1, -2))
      depth_utils.append(history_value)
      
      
      # TODO: This is dumb
      if player != 1:
        p1_value = jnp.sum(action_value * history_strategies[d][1, :, None, ...], axis=-1)
        p1_cf_regret = (p1_value - history_value[..., None]) * jnp.expand_dims(history_reaches[d][1], -1)
        p1_bin_regrets = jnp.bincount(self.constants.depth_history_actions[d][0].ravel(), p1_cf_regret.ravel(), length=self.constants.depth_actions[d] * self.constants.depth_iset_legal[d][0].shape[0]).reshape(regrets[d][0].shape) * self.constants.depth_iset_legal[d][0]
        
        cf_value = history_value[..., None] * jnp.expand_dims(history_reaches[d][1], -1) 
        bin_cf_value = jnp.bincount(self.constants.depth_history_iset[d][0].ravel(), cf_value.ravel(), length=self.constants.depth_iset_legal[d][0].shape[0]).reshape(cf_values[d][0].shape)
        
        bin_reaches = jnp.bincount(self.constants.depth_history_iset[d][0].ravel(), history_reaches[d][1].ravel(), length=self.constants.depth_iset_legal[d][0].shape[0]).reshape(cf_values[d][0].shape)
        
        bin_cf_value = jnp.where(bin_reaches > 1e-8, bin_cf_value / bin_reaches, bin_cf_value) 
        cf_values[d][0] = cf_values[d][0] + (bin_cf_value - cf_values[d][0]) * (2 / (iteration + 1))
        
        regrets[d][0] = jnp.maximum(regrets[d][0] + p1_bin_regrets, 0.0)
        
      if player != 0:
        p2_value = jnp.sum(action_value * history_strategies[d][0,..., None], axis=-2)
        p2_cf_regret = (p2_value - history_value[..., None]) * jnp.expand_dims(history_reaches[d][0], -1)
        p2_bin_regrets = jnp.bincount(self.constants.depth_history_actions[d][1].ravel(), p2_cf_regret.ravel(), length=self.constants.depth_actions[d] * self.constants.depth_iset_legal[d][1].shape[0]).reshape(regrets[d][1].shape) * self.constants.depth_iset_legal[d][1]
        
        cf_value = history_value[..., None] * jnp.expand_dims(history_reaches[d][0], -1) 
        
        bin_cf_value = jnp.bincount(self.constants.depth_history_iset[d][1].ravel(), cf_value.ravel(), length=self.constants.depth_iset_legal[d][1].shape[0]).reshape(cf_values[d][1].shape)
        
        bin_reaches = jnp.bincount(self.constants.depth_history_iset[d][1].ravel(), history_reaches[d][0].ravel(), length=self.constants.depth_iset_legal[d][1].shape[0]).reshape(cf_values[d][1].shape)
        
        bin_cf_value = jnp.where(bin_reaches > 1e-8, bin_cf_value / bin_reaches, bin_cf_value)        
        cf_values[d][1] = cf_values[d][1] +  (bin_cf_value - cf_values[d][1]) * (2/(iteration + 1))
        
        regrets[d][1] = jnp.maximum(regrets[d][1] - p2_bin_regrets, 0.0)
      
      
      # history_value = jnp.sum(action_value *)
    return regrets, averages, cf_values