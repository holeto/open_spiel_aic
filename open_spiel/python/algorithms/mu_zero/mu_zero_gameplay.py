
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
    self.mvs_actions = self.muzero.config.transformations + 1

    #DL Tree
    self.iset_map = [[], []]
    # TODO: Remove, these will be stored in MuZeroCFRConstants, that will be passed used to run CFR 

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
    return np.stack((p1_iset[None, ...], p2_iset[None, ...]), 0) # Shape would be [Pl, 1, Iset]
  
  def find_root_from_previous(self, public_state, iset):
  # We are passing public state and infoset separately, but from iset you should be able to get public state ideally.
    pass
    #End this method by clearing the previous tree
    # self.clear_tree()
  
  def check_iset_similarity(self, iset1, iset2):
    return False
   
  def validate_terminal(self, terminal, threshold: float = 0.5):
    return terminal < threshold
  
  # Starts in a single public state and creates a DL-tree.
  # Each layer should be done at once. Any call to NN should be done once!
  def prepare_cfr_structure(self, isets, reaches, cf_values, construct_gadget):
    chex.assert_equal(isets.shape[:-1], reaches.shape)
    chex.assert_equal(isets.shape[1:-1], cf_values.shape)
  
    dummy_iset = np.zeros_like(isets[0, 0])
    iset_map = [[dummy_iset], [dummy_iset]] # We create initial dummy iset 0
    
    depth_history_action_utility = []
    depth_history_iset = []
    depth_history_actions = []
    depth_history_legal = []
    depth_history_previous_iset = []
    depth_history_previous_action = []
    depth_history_previous_history = []
    depth_history_next_history = []
    # TODO: Do the same thing as with histories? Since we know that each player plays at each turn. 
    iset_previous_action = []
    iset_action_mask = []
    iset_action_depth = []
    
    # TODO: Split the map to be separate for each depth.
    def create_iset_map(curr_iset, amount_actions):
      isets = [[], []]
      for pl in range(curr_iset.shape[0]):
        first_iset_id = len(iset_map[pl])
        for i in range(curr_iset.shape[1]): 
          curr_index = -1
          for j in range(first_iset_id, len(iset_map[pl])):
            if self.check_iset_similarity(iset_map[pl][j], curr_iset[pl, i]):
              curr_index = j
              break
          if curr_index < 0:
            curr_index = len(iset_map[pl])
            iset_map[pl].append(curr_iset[pl, i])  
          isets[pl].append(curr_index)
          
      isets = np.array(isets)
      actions = isets[..., None] * amount_actions + np.arange(amount_actions)[None, None, ...] 
      return isets, actions
    
    
    def handle_mvs_layer(curr_iset, prev_iset, prev_action, prev_history):
      isets, actions = create_iset_map(curr_iset, self.mvs_actions)
      #print(isets.shape)
      mvs_vals = self.muzero.get_mvs_from_abstraction(curr_iset[0], curr_iset[1])
      #mvs values are from the perspective of player 0
      transformation_utils = np.stack((mvs_vals, -mvs_vals), axis=0)
      legal = np.ones((curr_iset.shape[1], 2, self.mvs_actions, self.mvs_actions), dtype=bool)
      next_history = np.full((curr_iset.shape[1], self.mvs_actions, self.mvs_actions), -1)
      
      depth_history_previous_iset.append(prev_iset)
      depth_history_previous_action.append(prev_action)
      depth_history_previous_history.append(prev_history)
      
      
      depth_history_action_utility.append(transformation_utils)
      depth_history_iset.append(isets)
      depth_history_actions.append(actions) 
      depth_history_legal.append(legal) 
      depth_history_next_history.append(next_history)
      
      
    
    
    def handle_single_layer(curr_iset, prev_iset, prev_action, prev_history, depth):
      isets, actions = create_iset_map(curr_iset, self.actions)
      # TODO: Could this be jitted from here onward?
      # What spedup would that bring? Would require to change some indexing to jnp.where 
      
      p1_legal, p2_legal = self.muzero.get_both_legal_actions_from_abstraction(curr_iset[0], curr_iset[1])
      p1_legal, p2_legal = p1_legal > 0, p2_legal > 0
      legal = p1_legal[..., None] * p2_legal #[..., None, :]
      # If we ever change to Bool[D, H(D),Pl, A], Instead of [D, H(D),A1, A2]
      # legal_stacked = np.stack((p1_legal, p2_legal), 0)
      
      
      # Even with in dimension -1, we want output dimension to be before the last dimension.
      # Checked that this does what is supposed (which is np.transpose(res, (0, 2, 3, 1)))
      vectorized_abstraction = jax.vmap(jax.vmap(self.muzero.get_next_state_from_abstraction, in_axes=(None, None, -1, -1), out_axes=(-2, -2, -2, -2)), in_axes=(None, None, -1, -1), out_axes=(-2, -2, -2, -2))
      
      # TODO: Can this be done better so we do not have to copy the actions for each player, but so that we can just use it as it is.
      p1_actions = np.repeat(np.arange(self.actions)[None, ...], self.actions, axis=0)
      p1_actions = np.repeat(p1_actions[None, ...], curr_iset.shape[1], axis= 0) # Repeats for each history
      #TODO: Is this correct? Will it not be always identical to p1_actions?
      p2_actions = np.transpose(p1_actions, (0, 2, 1))
      next_p1_isets, next_p2_isets, next_utilities, next_terminal = vectorized_abstraction(curr_iset[0], curr_iset[1], p1_actions, p2_actions) 
      
      action_utility = legal * next_utilities[..., 0] # We will select only utilities of player 0. We can do some more fancy stuff here, but whatever.
      # action_utility = legal * (next_utilities[..., 0] - next_utilities[..., 1]) / 2
      
      
      non_terminal = np.squeeze(self.validate_terminal(next_terminal), -1) * legal
      # next_flattened_p1_isets = np.choose()
      
      
      # From [H(D), A1, A2] should select [H(D + 1)] 
      # nonzero() returns indices which are non zero in tuple (4-tuple in this case)
      nonzeros = non_terminal.nonzero()
      next_isets = np.stack((next_p1_isets, next_p2_isets), 0)
      both_actions = np.stack((p1_actions, p2_actions), 0)
      next_isets = next_isets[:, *nonzeros, :] 
      
      
      # For each history from H(D + 1) select previous infoset and action
      next_prev_isets = isets[:, nonzeros[0]]
      next_prev_actions = both_actions[:, *nonzeros]
      next_prev_actions = next_prev_isets * self.actions + next_prev_actions
      
      # This should be easy, just for each nonzero terminal create value based on it's index in first dimension
      next_prev_history = nonzeros[0]
    
      
      # This should be -1 everywhere, except the part where you have next history. Therey ou go by terminal and just add 1
      next_history = (np.cumsum(non_terminal).reshape(non_terminal.shape) * non_terminal) - 1
      
      depth_history_action_utility.append(action_utility)
      depth_history_iset.append(isets)
      depth_history_actions.append(actions)
      depth_history_legal.append(legal) 
      depth_history_previous_iset.append(prev_iset)
      depth_history_previous_action.append(prev_action)
      depth_history_previous_history.append(prev_history)
      depth_history_next_history.append(next_history)
      
      if depth + 1 == self.config.depth_limit:
        handle_mvs_layer(next_isets,
                         next_prev_isets,
                         next_prev_actions,
                         next_prev_history
                         )
      else:
        handle_single_layer(next_isets, 
                            next_prev_isets,
                            next_prev_actions,
                            next_prev_history,
                            depth+1
                            )
      
      
    def handle_gadget_layer(curr_iset, prev_iset, prev_action, prev_history, cf_values):
       
      isets, actions = create_iset_map(curr_iset, 2) # Different amount of actions, only 2 for each player
      # TODO: Can these be done better?
      action_utilities = np.zeros((cf_values.shape[0], 2, 2))
      # Resolving player plays the only legal action, while the other terminates the game
      action_utilities[:, 0, 0] = cf_values #
      legals = np.ones((cf_values.shape[0], 2, 2))
      next_history = np.full((cf_values.shape[0], 2, 2), -1)
      if self.config.player == 0:
        legals[:, 1, :] = 0
        next_history[:, 0, 1] = np.arange(cf_values.shape[0])
      else:
        legals[:, :, 1] = 0 
        next_history[:, 1, 0] = np.arange(cf_values.shape[0])
      
      depth_history_iset.append(isets)
      depth_history_actions.append(actions)
      depth_history_action_utility.append(action_utilities)
      depth_history_legal.append(legals)
      depth_history_next_history.append(next_history)
      depth_history_previous_iset.append(prev_iset)
      depth_history_previous_action.append(prev_action)
      depth_history_previous_history.append(prev_history)
      
      next_prev_iset = isets
      next_prev_action = np.stack((next_prev_iset, next_prev_iset + 1), 1)
      handle_single_layer(curr_iset, isets, prev_action, prev_history, 0)   
       
    prev_iset = np.zeros_like(reaches, dtype=np.int64)
    prev_action = np.zeros_like(reaches, dtype=np.int64)
    prev_history = np.zeros_like(reaches, dtype=np.int64)
    
    if construct_gadget:
      handle_gadget_layer(isets, prev_iset, prev_action, prev_history, cf_values)
    else:
      handle_single_layer(isets, prev_iset, prev_action, prev_history, 0)

    return MuZeroCFRConstants()
    
 
  
  # def prepare_cfr_gadget_structure(self, p1_iset, p2_iset, reaches, cf_values):
  #   #dummy iset for the gadget
  #   dummy_iset = np.zeros_like(p1_iset[0])
  #   self.iset_map[0][0] = dummy_iset
  #   self.iset_map[1][0] = dummy_iset
  #   #FIXME: This is wrong, for each 
  #   # distinct iset in opponent isets
  #   #create a gadget
  #   isets = np.array([[0], [0]])
  #   actions = isets[..., None] * self.actions + np.arange(self.actions)[None, None, ...]
  #   #FIXME: Use size amount of isets for each player instead of 1
  #   opp_legals = np.zeros((1, self.actions), dtype=bool)
  #   pl_legals = np.zeros((1, self.actions), dtype=bool)
  #   pl_legals[0][0] = 1
  #   opp_legals[0][0] = 1
  #   opp_legals[0][1] = 1
  #   legals = opp_legals[None, ...] * pl_legals
  #   #making sure that player 1 actions are in the rows
  #   if self.config.player == 0:
  #     legals = np.transpose(legals, (0, 2, 1))

  #   #FIXME: Again instead of 1 number of isets for opponent
  #   non_terminal = np.ones((1, self.actions, self.actions))
  #   action_utilities = np.zeros((2, 1, self.actions, self.actions))
  #   #first action by both players corresponds to terminal nodes
  #   non_terminal[:, 0, 0] = 0
  #   #This assumes that counterfactual values are trained per infoset in CFR
  #   #and that they are given from perspective of acting player
  #   action_utilities[self.config.player, :, 0, 0] = cf_values
  #   action_utilities[1 - self.config.player, :, 0, 0] = -cf_values

  #   next_history = (np.cumsum(non_terminal).reshape(non_terminal.shape) * non_terminal) - 1
    
  #   self.depth_history_action_utility.append(action_utilities)
  #   self.depth_history_iset.append(isets)
  #   self.depth_history_actions.append(actions)
  #   self.depth_history_legal.append(legals)
  #   self.depth_history_previous_iset.append([np.zeros((1,))])
  #   self.depth_history_previous_action.append([np.zeros((1,))])
  #   self.depth_history_previous_history.append([np.zeros((1,))])
  #   self.depth_history_next_history.append(next_history)
  #   self.init_iset_reaches = reaches
  #   self.prepare_cfr_structure(p1_iset, p2_iset)
    
  #   return MuZeroCFRConstants()
    
  
  def run_cfr(self, cfr):
    pass  
   
  # Returns either policy or None. latter is that the policy is not computed yet.
  def get_policy(self, iset):
    return None
    
  def get_action(self, public_state, iset):
    optional_policy = self.get_policy(iset)
    if optional_policy is not None:
      return np.random.choice(self.actions, p=optional_policy)
    construct_gadget = not self.new_game
    if self.new_game:
      self.new_game = False
      isets = self.build_initial_root(public_state, iset)
      
      reaches = np.ones((2, isets.shape[1]))
      cf_values = np.zeros((isets.shape[1],))
    else:
      isets, reaches, cf_values= self.find_root_from_previous(public_state, iset) 
      
    cfr = self.prepare_cfr_structure(isets, reaches, cf_values, construct_gadget)
    
    policy = self.run_cfr(cfr)
    return np.random.choice(self.actions, p=policy)
  
  
  