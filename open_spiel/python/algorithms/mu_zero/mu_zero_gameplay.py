
import jax
import jax.numpy as jnp
import jax.lax as lax

import chex
import numpy as np

from open_spiel.python.algorithms.mu_zero.jax_goofspiel import JaxOriginalGoofspiel
from open_spiel.python.algorithms.mu_zero.mu_zero import MuZeroTrain


@chex.dataclass(frozen=True)
class MuZeroGameplayConfig:
  player: int = 0
  resolve_iterations: int = 1000
  depth_limit: int = 1


@chex.dataclass(frozen=True)
class MuZeroCFRConstants:
  """Constants for JaxCFR."""
 
  max_depth: int 
  max_actions: int

  max_iset_depth: chex.ArrayTree = ()  # Is just a list of integers
  isets: chex.ArrayTree = ()  # Is just a list of integers
  
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
  def __init__(self):
    pass

# The main idea is:
# Create root
class MuZeroGameplay:
  def __init__(self, muzero: MuZeroTrain, config: MuZeroGameplayConfig) -> None:
    self.config = config
    self.muzero = muzero
    self.actions = muzero.actions

    self.new_game = True # flag that specifies whether we are at the beginning of the game or whether we have moved
    
    self.init_isets = self.initialize_isets()
    
  def initialize_isets(self):
    if isinstance(self.muzero.game, JaxOriginalGoofspiel):
      init_info = self.muzero.game.initialize_structures()[:-1] # Last thing is a legal actions
      _, p1_iset, p2_iset, _ = self.muzero.game.get_info(*init_info)
      return p1_iset, p2_iset
    else:
      state = self.muzero.game.new_initial_state()
      return np.array(state.information_state_tensor(0)), np.array(state.information_state_tensor(1))
    
  def reset(self):
    self.new_game = True
  
  # TODO: pass the state?
  # TODO: Do we need this method? Maybe we could just do this in initilize_isets
  # TODO: Shouldn't we just p1_iset and p2_iset in a single array?
  # THIS IS THE ONLY PART WHERE WE USE OPPONENT'S INFOSET! It is because at the beginning of the game both players know the state exactly. We cannot use this knowledge anywhere else
  def build_initial_root(self, public_state, iset):
    assert np.allclose(iset, self.init_isets[self.config.player])
    p1_iset, p2_iset = self.muzero.get_both_abstraction(public_state, *self.init_isets)
    return p1_iset[None, ...], p2_iset[None, ...]
    
  def find_root_from_previous(self, public_state, iset):
  # We are passing public state and infoset separately, but from iset you should be able to get public state ideally.
    pass
  
  def check_iset_similarity(self, iset1, iset2):
    return False
   
  def validate_terminal(self, terminal, threshold: float = 0.5):
    return terminal < threshold

    
  # Starts in a single public state and creates a DL-tree.
  # Each layer should be done at once. Any call to NN should be done once!
  def prepare_cfr_structure(self, p1_iset, p2_iset):
    assert len(p1_iset) == len(p2_iset) 
    iset_map = [[], []]
    depth_history_action_utility = []
    depth_history_iset = []
    depth_history_actions = []
    depth_history_legal = []
    depth_history_previous_iset = []
    depth_history_previous_action = []
    depth_history_previous_history = []
    depth_history_next_history = []
    
    def handle_single_layer(p1_iset, p2_iset, prev_iset, prev_action, prev_history):
      action_utility = []
      isets = [[], []]
      actions = []
      legal = [] 
      next_history = []
      p1_legal, p2_legal = self.muzero.get_both_legal_actions_from_abstraction(p1_iset, p2_iset)
      p1_legal, p2_legal = p1_legal > 0, p2_legal > 0
      legal_stacked = np.stack((p1_legal, p2_legal), 0)
      # TODO: verify this is correct
      legal_prod = p1_legal[..., None] * p2_legal[..., None, :]
      legal = legal_prod
      # iset = np.stack((p1_iset, p2_iset), 0)
  
      # TODO: Do not check the whole map, but only the part of the map in the current depth
      # So probably initialize the index where to start as len(iset_map[pl]).
      for pl, pl_isets in enumerate([p1_iset, p2_iset]):
        for i, iset in enumerate(pl_isets):
          curr_index = -1
          for j, im in enumerate(iset_map[pl]):
            if self.check_iset_similarity(im, iset):
              curr_index = j
              break
          if curr_index < 0:
            curr_index = len(iset_map[pl])
            iset_map[pl].append(iset)  
          isets[pl].append(curr_index)
      
      isets = np.array(isets)
      action = isets[..., None] * self.actions + np.arange(self.actions)[None, None, ...] 
      vectorized_abstraction = jax.vmap(jax.vmap(self.muzero.get_next_state_from_abstraction, in_axes=(None, None, -1, -1), out_axes=(-1, -1, -1, -1)), in_axes=(None, None, -1, -1), out_axes=(-1, -1, -1, -1))
      
      # TODO: Can this be done better so we do not have to copy the actions for each player, but so that we can just use it as it is.
      p1_actions = np.repeat(np.arange(self.actions)[None, ...], self.actions, axis=0)
      p1_actions = np.repeat(p1_actions[None, ...], len(p1_iset), axis= 0) # Repeats for each history
      p2_actions = np.transpose(p1_actions, (0, 2, 1))
      next_p1_isets, next_p2_isets, next_utilities, next_terminal = vectorized_abstraction(p1_iset, p2_iset, p1_actions, p2_actions) 
      action_utility = next_utilities
      non_terminal = self.validate_terminal(next_terminal) 
      # next_flattened_p1_isets = np.choose()
      # FIXME: This seems bad, but working
      # From [H(D), A1, A2] should select [H(D + 1)] 
      # nonzero() returns indices which are non zero in tuple (4-tuple in this case)
      nonzeros = non_terminal.nonzero()
      next_flattened_p1_isets = next_p1_isets[nonzeros[0], :, nonzeros[2], nonzeros[3]]
      next_flattened_p2_isets = next_p2_isets[nonzeros[0], :, nonzeros[2], nonzeros[3]]
      
      # TODO: Until here in first iteration it seems to work
      
      # FIXME: This will not work
      # For each history from H(D + 1) select previous infoset and action 
      next_prev_p1_isets = iset[0][non_terminal.nonzero()]
      next_prev_p2_isets = iset[1][non_terminal.nonzero()]
      next_prev_actions_p1 = p1_actions[non_terminal.nonzero()]
      next_prev_actions_p2 = p2_actions[non_terminal.nonzero()]
      next_prev_actions_p1 = next_prev_p1_isets * self.actions + next_prev_actions_p1
      next_prev_actions_p2 = next_prev_p2_isets * self.actions + next_prev_actions_p2
      
      # FIXME
      # This should be easy, just for each nonzero terminal create value based on it;s index in first dimension
      next_prev_history = terminals.nonzero()[0]
    
      #FIXME
      # This should be -1 everywhere, except the part where you have net history. Ther eyou go by terminal and just add 1
      next_history = (np.cumsum(next_terminal).reshape(next_terminal.shape) * next_terminal) - 1
      
      depth_history_action_utility.append(action_utility)
      depth_history_iset.append(isets)
      depth_history_actions.append(actions)
      depth_history_legal.append(legal)
      depth_history_previous_iset.append(prev_iset)
      depth_history_previous_action.append(prev_action)
      depth_history_previous_history.append(prev_history)
      depth_history_next_history.append(next_history)
      
      handle_single_layer(next_flattened_p1_isets, next_flattened_p2_isets, 
                          (next_prev_p1_isets, next_prev_p2_isets),
                          (next_prev_actions_p1, next_prev_actions_p2),
                          next_prev_history
                          )
      
    handle_single_layer(p1_iset, p2_iset, np.zeros((1,)),  np.zeros((1,)),  np.zeros((1,)))
 
  
  def prepare_cfr_gadget_structure(self, p1_iset, p2_iset, reaches, cf_values):
    pass
  
  def run_cfr(self, cfr):
    pass  
   
  # Returns either policy or None. latter is that the policy is not computed yet.
  def get_policy(self, iset):
    return None
    
  def get_action(self, public_state, iset):
    optional_policy = self.get_policy(iset)
    if optional_policy is not None:
      return np.choice(self.actions, p=optional_policy)
    if self.new_game:
      self.new_game = False
    
      p1_iset, p2_iset = self.build_initial_root(public_state, iset)  
      cfr = self.prepare_cfr_structure(p1_iset, p2_iset)
      
    else:
      p1_iset, p2_iset, reaches, cf_values= self.find_root_from_previous(public_state, iset)
      cfr = self.prepare_cfr_gadget_structure(p1_iset, p2_iset, reaches, cf_values)
      
    policy = self.run_cfr(cfr)
    return np.choice(self.actions, p=policy)
  
  
  