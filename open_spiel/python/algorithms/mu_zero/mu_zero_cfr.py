
import chex
import jax
import jax.numpy as jnp
from open_spiel.python.jax.cfr.jax_cfr import regret_matching, update_regrets_plus, JAX_CFR_SIMULTANEOUS_UPDATE

@chex.dataclass(frozen=True)
class MuZeroCFRConstants:
  """Constants for JaxCFR."""
 
  max_depth: int 
  max_actions: int
  non_gadget_root_depth: int #At which depth is the root excluding gadget
  resolving_player: int

  max_iset_depth: chex.ArrayTree = ()  # Is just a list of integers
  depth_actions: chex.ArrayTree = ()  # Is just a list of integers
  depth_iset_map: chex.ArrayTree = () # ID -> Abstract iset for each D [D, Pl, S(D)]
  
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
  #init iset reaches assumed to be S(D) of the root 
  def __init__(self, constants: MuZeroCFRConstants, init_iset_reaches):
    self.constants = constants
    self.players = 2
    self._linear_averaging = True
    self._regret_matching_plus = True
    self._alternating_updates = True

    self.update_regrets = update_regrets_plus
    self.regret_matching = regret_matching
    self.init_iset_reaches = init_iset_reaches[self.constants.depth_history_iset[self.constants.non_gadget_root_depth, self.constants.resolving_player, :]]

    
    #Should be [D, Pl, S(D), A] ?
    self.regrets = [[jnp.zeros((self.constants.iset_previous_action[d][pl].shape[0], a)) for pl in range(2)] for d, a in enumerate(constants.depth_actions)]
    #Should be [D, Pl, S(D), A] ?   
    self.averages = [[jnp.zeros((self.constants.iset_previous_action[d][pl].shape[0], a)) for pl in range(2)] for d, a in enumerate(constants.depth_actions)]
    #TODO: Making assumption that regrets and averages will be converted to a single jnp array instead of lists of jnp arrays
    #not sure if it is correct.

  def propagate_strategy(self, current_strategy):
    #Expecting current strategy to be [D, Pl, S(D), A]
    realization_plans = jnp.ones_like(current_strategy[pl])
    for pl in range(self.players):
      for depth in range(1, self.constants.max_depth + 1):
        realization_plans[pl, depth, ...] = realization_plans[pl, depth - 1, ...] * current_strategy[pl, depth - 1, :, self.constants.iset_previous_action]
    return realization_plans
  
  def average_policy_dict(self, stop_depth = -1):
    stop = stop_depth if stop_depth > 0 else self.constants.max_depth
    average_dict = {}
    for depth in range(self.constants.non_gadget_root_depth, stop):
      for iset_idx in range(self.averages[depth, self.constants.resolving_player].shape[0]):
        iset = self.constants.depth_iset_map[depth, self.constants.resolving_player, iset_idx]
        iset_str = jnp.array_str(iset)
        if not iset_str in average_dict:
          average_dict[iset_str] = self.averages[depth, self.constants.resolving_player, iset_idx, :]
    return average_dict
  
  def average_root_policy_dict(self):
    return self.average_policy_dict(stop_depth=1)
  
  def step(self, regrets, averages, average_policy_update_coefficient, player):
    #[D, Pl, S(D), A]
    current_strategies = self.regret_matching(self.regrets)
    #weight the strategies using initial iset reaches
    weighted_strategies = jnp.copy(current_strategies)
    weighted_strategies[self.constants.non_gadget_root_depth, self.resolving_player, ...] *= self.init_iset_reaches
    #[D, Pl, S(D), A]
    realization_plans = self.propagate_strategy(weighted_strategies)
    #[D, Pl, S(D)]
    iset_reaches = jnp.sum(realization_plans, axis=-1)
    #propagate from down to top
    #[H(D), A(1 -pl), A(pl)]
    depth_action_utils = self.constants.depth_history_action_utility[-1, ...]
    for d in range(self.constants.max_depth - 1, -1, -1):
      for pl in range(self.players):
        opp = 1 - pl
        #Transpose, so that opponent is always the row player
        if pl == 0:
          depth_action_utils = jnp.transpose(depth_action_utils, (0, 2, 1))
        depth_action_utils = (depth_action_utils * self.constants.depth_history_legal[d]) *  (1 - 2 * pl)
        #weight the rows by the opponent current policy and sum them to get action value
        #[H(D), A(pl)]
        action_value = jnp.sum(depth_action_utils * current_strategies[d][opp][self.constants.depth_history_iset[d, opp, :]][..., jnp.newaxis], axis=1)
        #[H(D), 1]
        history_value = jnp.sum(action_value * current_strategies[d,pl, self.constants.depth_history_iset[d, pl, :], :])
        #[H(D), A(pl)]
        regret = (action_value - history_value[..., jnp.newaxis])
        #TODO: Make sure to avoid resolving gadget root for the resolving player 

        #TODO: Add the lenght parameter, because jit needs it
        #[S(D), A(pl)]
        bin_regret = jnp.bincount(self.constants.depth_history_actions[d, pl, ...].ravel(), regret.ravel())
        bin_regret = bin_regret * realization_plans[d, pl, ...]
        if d > 0:
          #FIXME: Indexing it like this probably will not work
          depth_action_utils = history_value[self.constants.depth_history_next_history[d - 1, ...]]
        regrets[d][pl] = jnp.where(jnp.logical_or(player == pl, player == JAX_CFR_SIMULTANEOUS_UPDATE), regrets[d][pl] + bin_regret)
    #FIXME: To work like this, regrets and averages  would have to be stored in one jnp array. Do we want that?
    regrets = self.update_regrets(regrets)
    averages = jnp.where(jnp.logical_or(player == pl, player == JAX_CFR_SIMULTANEOUS_UPDATE), averages + current_strategies  * iset_reaches[..., jnp.newaxis]  * average_policy_update_coefficient, averages)
    #TODO: Add counterfactual values
    return regrets, averages
  
  def multiple_steps(self, num_steps):
    for i in range(num_steps):
      averaging_coefficient = i + 1 if self._linear_averaging else 1
      if(self._alternating_updates):
        for pl in range(self.players):
          self.regrets, self.averages = self.step(self.regrets, self.averages, averaging_coefficient, pl)
      else:
          self.regrets, self.averages = self.step(self.regrets, self.averages, averaging_coefficient, JAX_CFR_SIMULTANEOUS_UPDATE)


    
