
import jax
import jax.numpy as jnp
import jax.lax as lax

import chex
import numpy as np
import queue

from open_spiel.python.algorithms.mu_zero.jax_goofspiel import JaxOriginalGoofspiel
from open_spiel.python.algorithms.mu_zero.mu_zero import MuZeroTrain
from open_spiel.python.algorithms.mu_zero.mu_zero_cfr import MuZeroCFRConstants, MuZeroCFR, check_iset_similarity

@chex.dataclass(frozen=True)
class MuZeroGameplayConfig:
  player: int = 0
  resolve_iterations: int = 1000
  depth_limit: int = 1


def convert_player_depth_to_jax(arr):
  return [[jnp.array(p) for p in d] for d in arr]

def convert_depth_to_jax(arr):
  return [jnp.array(d) for d in arr]

# The main idea is:
# Create root
class MuZeroGameplay:
  def __init__(self, muzero: MuZeroTrain, config: MuZeroGameplayConfig) -> None:
    self.config = config
    self.muzero = muzero
    self.actions = muzero.actions
    self.mvs_actions = self.muzero.config.transformations + 1
    self.new_game = True # flag that specifies whether we are at the beginning of the game or whether we have moved
    self.initialize_isets() 
    
    self.cfr = None
    self.policy = {}
    
  # First finds the information states and public states from the game, then pushes them through abstraction layer
  def initialize_isets(self):
    if isinstance(self.muzero.game, JaxOriginalGoofspiel):
      init_info = self.muzero.game.initialize_structures()[:-1] # Last thing is a legal actions
      _, *self.init_info = self.muzero.game.get_info(*init_info) 
    else:
      state = self.muzero.game.new_initial_state()
      self.init_info =  np.array(state.information_state_tensor(0)), np.array(state.information_state_tensor(1)), np.array(state.public_state_tensor())
    p1_iset, p2_iset = self.muzero.get_both_abstraction(self.init_info[2], self.init_info[0], self.init_info[1])
    self.init_iset = np.stack((p1_iset[None, ...], p2_iset[None, ...]), 0) # Shape would be [Pl, 1, Iset]
    
    
  def reset(self):
    self.new_game = True 
    
  # TODO: pass the state?
  # TODO: Do we need this method? Maybe we could just do this in initilize_isets
  # TODO: Shouldn't we just p1_iset and p2_iset in a single array?
  # THIS IS THE ONLY PART WHERE WE USE OPPONENT'S INFOSET! It is because at the beginning of the game both players know the state exactly. We cannot use this knowledge anywhere else
  def build_initial_root(self, public_state, iset):
    assert np.allclose(iset, self.init_info[self.config.player])
    return self.init_iset
   

  def find_root_from_previous(self, public_state, iset):
  # We are passing public state and infoset separately, but from iset you should be able to get public state ideally.
    #interested in the reaches for the resolving player in the last layer
    last_layer_CF_vals = self.cfr.get_last_depth_player_cf_values(self.config.player)
    #abstracted_iset = self.muzero.get_abstraction(public_state, iset, self.config.player)
    #TODO: Propagate the current reaches in CFR
    # and return per history reaches
    reaches = self.cfr.find_reaches_from_average()
    #interested in the reaches for the resolving player in the last layer
    #[S(D)]
    last_layer_reaches = reaches[-1, self.config.player, :]
    #[Pl,H(D)]
    last_layer_iset_indices = jnp.stack([self.cfr.constants.depth_history_iset[-1][pl] for pl in range(2)], axis=0)
    #Assuming equal number of isets for both player
    num_isets = last_layer_iset_indices[self.config.player].shape[0]
    #[Pl,H(D)]
    last_layer_isets = jnp.stack([self.cfr.constants.depth_iset_map[-1][pl][last_layer_iset_indices[pl]] for pl in range(2)], axis=0)
    #The version using public state decoder
    vectorized_decoder = jax.vmap(self.muzero.get_decoded_public_state, in_axes=(0, None), out_axes=0)
    vectorized_compare = jax.vmap(jax.vmap(check_iset_similarity, in_axes=(0, None), out_axes=0), in_axes=(0, None), out_axes=0)
    last_layer_pub_states = []
    for pl in range(2):
      last_layer_pub_states.append(vectorized_decoder(last_layer_isets[pl], pl))
    last_layer_pub_states = jnp.stack(last_layer_pub_states, axis=0)
    #Should be [Pl, H(D)]
    pub_state_mask_pl = vectorized_compare(last_layer_pub_states, public_state)
    #[H(D)]
    pub_state_mask = jnp.logical_and(pub_state_mask_pl[0], pub_state_mask_pl[1]).flatten()
    found_node_indices = pub_state_mask.nonzero()
    stacked_nodes = jnp.stack([last_layer_isets[0][found_node_indices], last_layer_isets[1][found_node_indices]], axis = 0)
    #TODO: Returning it like this assumes that the reaches have shape H(D)
    # and returns reaches per history. Would that be a problem?
    #TODO: Add CF values to the CFR and return them
    return stacked_nodes[:, None, :], last_layer_reaches[found_node_indices], last_layer_CF_vals[found_node_indices]
    #This is a version without using the public state decoder
    #TODO: Naive version
    # pub_state = []
    #Find idx of the initial iset
    # start_iset_idx = 0
    # for i in range(num_isets):
    #     pl_iset = last_layer_isets[self.config.player][i]
    #     if check_iset_similarity(pl_iset, iset):
    #       start_iset_idx = last_layer_iset_indices[self.config.player][i]
    #       #start_iset_idx = i
    #       break
    # visited = [[], []]
    # q = queue()
    # def check_visited(new_iset_idx, player):
    #   for iset_idx in visited[player]:
    #     if iset_idx == new_iset_idx:
    #       return True
    #   return False
    # def add_by_iset(iset_to_add, player):
    #   for i in range(num_isets):
    #     pl_iset = last_layer_isets[player][i]
    #     if check_iset_similarity(pl_iset, iset_to_add):
    #       opp_iset = last_layer_isets[1 - player][i]
    #       state = []
    #       if player == 0:
    #         state.append(pl_iset)
    #         state.append(opp_iset)
    #       else:
    #         state.append(opp_iset)
    #         state.append(pl_iset)
    #       opp_iset_idx = last_layer_iset_indices[1 - player][i]
    #       q.enqueue((opp_iset_idx, opp_iset, 1 - player))
    #       public_state.append(state)
    #   return q
    # q = add_by_iset(iset, self.config.player)
    # visited[self.config.player].append(start_iset_idx)
    # while not q.empty():
    #   opp_iset_idx, opp_iset, player = q.dequeue()
    #   if not check_visited(opp_iset_idx, player):
    #     add_by_iset(opp_iset, 0)
    #     visited[player].append(iset)
      
    #TODO: Pick only last layer reaches which are for isets in public state
    #TODO: Return also CF values from the CFR
    #return pub_state, last_layer_reaches[visited[self.config.player]]

  
  
  # def check_pub_state_intersection(self, iset, public_state):
  #   return False
   
  def validate_terminal(self, terminal, threshold: float = 0.5):
    return terminal < threshold
  
  # Starts in a single public state and creates a DL-tree.
  # Each layer should be done at once. Any call to NN should be done once!
  def prepare_cfr_structure(self, isets, reaches, cf_values, construct_gadget):
    chex.assert_equal(isets.shape[:-1], reaches.shape)
    chex.assert_equal(isets.shape[1:-1], cf_values.shape)
   
    depth_iset_map = [] # We create initial dummy iset 0
    depth_iset_legal = []
    
    depth_history_action_utility = []
    depth_history_iset = []
    depth_history_actions = []
    depth_history_legal = []
    
    depth_history_next_history = []
    # TODO: Do the same thing as with histories? Since we know that each player plays at each turn. 
    
    # TODO: Split the map to be separate for each depth.
    # Because of imperfect recall it does not make sense to have all the isets in the same map.
    def create_iset_map(curr_iset, amount_actions):
      isets = [[], []]
      iset_map = [[], []]
      for pl in range(curr_iset.shape[0]):
        first_iset_id = len(iset_map[pl])
        for i in range(curr_iset.shape[1]): 
          curr_index = -1
          for j in range(first_iset_id, len(iset_map[pl])):
            if check_iset_similarity(iset_map[pl][j], curr_iset[pl, i]):
              curr_index = j
              break
          if curr_index < 0:
            curr_index = len(iset_map[pl])
            iset_map[pl].append(curr_iset[pl, i])  
          isets[pl].append(curr_index)
          
      isets = np.array(isets)
      actions = isets[..., None] * amount_actions + np.arange(amount_actions)[None, None, ...] 
      iset_map = [np.array(i) for i in iset_map]
      return iset_map, isets, actions
    
    
    def handle_mvs_layer(curr_iset):
      iset_map, isets, actions = create_iset_map(curr_iset, self.mvs_actions)
      
      mvs_vals = self.muzero.get_mvs_from_abstraction(curr_iset[0], curr_iset[1])
      iset_legal = [np.ones(iset_map[pl].shape[:-1] + (self.mvs_actions,)) for pl in range(2)] 
      legal = np.ones_like(mvs_vals)
      next_history = np.full_like(mvs_vals, -1, dtype=int)
      
      depth_iset_map.append(iset_map)
      depth_iset_legal.append(iset_legal)
      
      depth_history_action_utility.append(mvs_vals)
      depth_history_iset.append(isets)
      depth_history_actions.append(actions) 
      depth_history_legal.append(legal) 
      depth_history_next_history.append(next_history)
      
      
    
    
    def handle_single_layer(curr_iset, depth):
      iset_map, isets, actions = create_iset_map(curr_iset, self.actions)
      # TODO: Could this be jitted from here onward?
      # What spedup would that bring? Would require to change some indexing to jnp.where 
      p1_legal_iset, p2_legal_iset = self.muzero.get_both_legal_actions_from_abstraction(iset_map[0], iset_map[1])
      p1_legal_iset, p2_legal_iset = p1_legal_iset > 0, p2_legal_iset > 0
      
      p1_legal, p2_legal = p1_legal_iset[isets[0]], p2_legal_iset[isets[1]] 
      iset_legal = [p1_legal_iset, p2_legal_iset]
      legal = p1_legal[..., None] * p2_legal[..., None, :]
      # If we ever change to Bool[D, H(D),Pl, A], Instead of [D, H(D),A1, A2]
      # legal_stacked = np.stack((p1_legal, p2_legal), 0)
      
      
      # Even with in dimension -1, we want output dimension to be before the last dimension.
      # Checked that this does what is supposed (which is np.transpose(res, (0, 2, 3, 1)))
      vectorized_abstraction = jax.vmap(jax.vmap(self.muzero.get_next_state_from_abstraction, in_axes=(None, None, -1, -1), out_axes=(-2, -2, -2, -2)), in_axes=(None, None, -1, -1), out_axes=(-2, -2, -2, -2))
      
      # TODO: Can this be done better so we do not have to copy the actions for each player, but so that we can just use it as it is.
      
      p2_actions = np.tile(np.arange(self.actions), (curr_iset.shape[1], self.actions, 1)) 
      p1_actions = np.transpose(p2_actions, (0, 2, 1))
      next_p1_isets, next_p2_isets, next_utilities, next_terminal = vectorized_abstraction(curr_iset[0], curr_iset[1], p1_actions, p2_actions) 
      
      action_utility = legal * next_utilities[..., 0] # We will select only utilities of player 0. We can do some more fancy stuff here, but whatever.
      # action_utility = legal * (next_utilities[..., 0] - next_utilities[..., 1]) / 2
      
      
      non_terminal = np.squeeze(self.validate_terminal(next_terminal), -1) * legal
      # next_flattened_p1_isets = np.choose()
      
      
      # From [H(D), A1, A2] should select [H(D + 1)] 
      # nonzero() returns indices which are non zero in tuple (4-tuple in this case)
      nonzeros = non_terminal.nonzero()
      next_isets = np.stack((next_p1_isets, next_p2_isets), 0)
      next_isets = next_isets[:, *nonzeros, :] 
      
      # This should be -1 everywhere, except the part where you have next history. Therey ou go by terminal and just add 1
      next_history = (np.cumsum(non_terminal).reshape(non_terminal.shape) * non_terminal) - 1
      
      depth_iset_map.append(iset_map)
      depth_iset_legal.append(iset_legal)
      depth_history_action_utility.append(action_utility)
      depth_history_iset.append(isets)
      depth_history_actions.append(actions)
      depth_history_legal.append(legal)  
      depth_history_next_history.append(next_history.astype(int))
      
      if np.all(next_history < 0):
        return
      
      if depth + 1 == self.config.depth_limit:
        handle_mvs_layer(next_isets)
      else:
        handle_single_layer(next_isets, depth+1)
      
      
    def handle_gadget_layer(curr_iset, cf_values):
       
      iset_map, isets, actions = create_iset_map(curr_iset, 2) # Different amount of actions, only 2 for each player
      # TODO: Can these be done better?
      
      action_utilities = np.zeros((cf_values.shape[0], 2, 2))
      # Resolving player plays the only legal action, while the other terminates the game
      action_utilities[:, 0, 0] = cf_values #
      iset_legal = [np.ones(iset_map[pl].shape[:-1] + (2,)) for pl in range(2)] 
      iset_legal[self.config.player][:, 1] = 0
      legals = np.ones((cf_values.shape[0], 2, 2))
      next_history = np.full((cf_values.shape[0], 2, 2), -1, dtype = int)
      if self.config.player == 0:
        legals[:, 1, :] = 0
        
        next_history[:, 0, 1] = np.arange(cf_values.shape[0])
      else:
        legals[:, :, 1] = 0 
        next_history[:, 1, 0] = np.arange(cf_values.shape[0])
      
      depth_iset_map.append(iset_map)
      depth_iset_legal.append(iset_legal)
      depth_history_iset.append(isets)
      depth_history_actions.append(actions)
      depth_history_action_utility.append(action_utilities)
      depth_history_legal.append(legals)
      depth_history_next_history.append(next_history)
      
      handle_single_layer(curr_iset, 0)   
       
    
    if construct_gadget:
      handle_gadget_layer(isets, cf_values)
    else:
      handle_single_layer(isets, 0)

    init_reaches = jnp.copy(reaches)
    init_condition = jnp.array([self.config.player == 0, self.config.player == 1])
    init_reaches = jnp.where(init_condition[..., None], init_reaches, 1) 
    
    constants = MuZeroCFRConstants(
      max_depth = len(depth_history_iset),
      resolving_player = self.config.player,
      
      init_reaches = init_reaches,
      
      depth_actions = [a.shape[-1] for a in depth_history_actions],
      depth_iset_map = convert_player_depth_to_jax(depth_iset_map),
      depth_iset_legal = convert_player_depth_to_jax(depth_iset_legal),
      
      depth_history_action_utility = convert_depth_to_jax(depth_history_action_utility),
      depth_history_iset = convert_depth_to_jax(depth_history_iset),
      depth_history_actions = convert_depth_to_jax(depth_history_actions),
      depth_history_legal = convert_depth_to_jax(depth_history_legal),
      
      depth_history_next_history = convert_depth_to_jax(depth_history_next_history),
    )
    cfr = MuZeroCFR(constants)
    
    return cfr
    

  def run_cfr(self, cfr):
    cfr.multiple_steps(self.config.resolve_iterations)

  def get_policy(self, iset):
    if self.cfr is None:
      return None
    return self.cfr.get_strategy(iset, self.config.player)

  def get_action(self, public_state, iset):
    
    abstracted_iset = self.muzero.get_abstraction(public_state, iset, self.config.player)
    optional_policy = self.get_policy(iset)
    if optional_policy is not None:
      return np.random.choice(self.actions, p=optional_policy)
    construct_gadget = not self.new_game
    # construct_gadget = True
    if self.new_game:
      self.new_game = False
      isets = self.build_initial_root(public_state, iset)
      
      reaches = np.ones((2, isets.shape[1]))
      cf_values = np.zeros((isets.shape[1],))
    else:
      isets, reaches, cf_values= self.find_root_from_previous(public_state, iset)
      
    # TODO: Refactor this.
    cfr = self.prepare_cfr_structure(isets, reaches, cf_values, construct_gadget)
    self.run_cfr(cfr)
    policy = cfr.get_strategy(abstracted_iset, self.config.player)
    
    self.cfr = cfr
    policy = np.asarray(policy, dtype="float64")
    policy /= np.sum(policy)
    #just in case
    self.policy[jnp.array_str(iset)] = policy
    return np.random.choice(self.actions, p=policy)
  
  
  