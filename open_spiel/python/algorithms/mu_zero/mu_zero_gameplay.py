
import jax
import jax.numpy as jnp
import jax.lax as lax

import chex
import numpy as np


from open_spiel.python.algorithms.mu_zero.jax_games.jax_goofspiel import JaxGoofspiel
from open_spiel.python.algorithms.mu_zero.mu_zero_train import MuZeroTrain
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

def validate_terminal(terminal, threshold: float = 0.5):
  return terminal < threshold

def find_next_root(cfr: MuZeroCFR, tree_depth: int, player: int, public_state, iset):
  opponent = 1 - player
  public_state_histories = cfr.find_public_state_from_iset(iset, player, tree_depth)
  history_reaches = cfr.find_reaches_from_average()[tree_depth][:, public_state_histories]
  # history_reaches = np.asarray(cfr.last_depth_reaches)[:, public_state_histories]
  # TODO: Use numpy or jax.numpy?
  next_reaches = np.where(np.array([[player == 0], [player == 1]]), history_reaches, 1.0)
  depth_isets = np.array(cfr.constants.depth_history_iset[tree_depth])
  depth_cf_vals = np.array(cfr.cf_values[tree_depth][opponent])
  next_isets_id = depth_isets[:, public_state_histories]
  next_cf_values = depth_cf_vals[next_isets_id[opponent]]
  next_isets = cfr.depth_iset_map[tree_depth][opponent][next_isets_id[opponent]]
  next_isets = np.stack([np.array(cfr.depth_iset_map[tree_depth][pl])[next_isets_id[pl]] for pl in range(2)], axis = 0)
  return next_isets, next_reaches, next_cf_values 


# Starts in a single public state and creates a DL-tree.
# Each layer should be done at once. Any call to NN should be done once!
def prepare_cfr_structure(muzero: MuZeroTrain, player: int, depth_limit, isets, reaches, cf_values, construct_gadget):
  chex.assert_equal(isets.shape[:-1], reaches.shape)
  chex.assert_equal(isets.shape[1:-1], cf_values.shape)
  mvs_actions = muzero.config.transformations + 1
  
  depth_iset_map = [] # We create initial dummy iset 0
  depth_iset_legal = []
  
  depth_history_action_utility = []
  depth_history_iset = []
  depth_history_actions = []
  depth_history_legal = []
  
  depth_history_next_history = [] 
  
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
    iset_map, isets, actions = create_iset_map(curr_iset, mvs_actions)
    
    mvs_vals = muzero.get_mvs_from_abstraction(curr_iset[0], curr_iset[1])
    iset_legal = [np.ones(iset_map[pl].shape[:-1] + (mvs_actions,)) for pl in range(2)]  
    legal = np.ones_like(mvs_vals)
    next_history = np.full_like(mvs_vals, -1, dtype=int)
    
    depth_iset_map.append(iset_map)
    depth_iset_legal.append(iset_legal)


    #jax.debug.breakpoint()
    depth_history_action_utility.append(mvs_vals)
    depth_history_iset.append(isets)
    depth_history_actions.append(actions) 
    depth_history_legal.append(legal) 
    depth_history_next_history.append(next_history)
    
  
  def handle_single_layer(curr_iset, depth):
    iset_map, isets, actions = create_iset_map(curr_iset, muzero.actions)
    # TODO: Could this be jitted from here onward?
    # What spedup would that bring? Would require to change some indexing to jnp.where 
    p1_legal_iset, p2_legal_iset = muzero.get_both_legal_actions_from_abstraction(iset_map[0], iset_map[1])
    p1_legal_iset, p2_legal_iset = p1_legal_iset > 0, p2_legal_iset > 0
    
    p1_legal, p2_legal = p1_legal_iset[isets[0]], p2_legal_iset[isets[1]]
    iset_legal = [p1_legal_iset, p2_legal_iset]
    legal = p1_legal[..., None] * p2_legal[..., None, :]
    # If we ever change to Bool[D, H(D),Pl, A], Instead of [D, H(D),A1, A2]
    # legal_stacked = np.stack((p1_legal, p2_legal), 0)
    
    
    # Even with in dimension -1, we want output dimension to be before the last dimension.
    # Checked that this does what is supposed (which is np.transpose(res, (0, 2, 3, 1)))
    vectorized_abstraction = jax.vmap(jax.vmap(muzero.get_next_state_from_abstraction, in_axes=(None, None, -1, -1), out_axes=(-2, -2, -2, -2)), in_axes=(None, None, -1, -1), out_axes=(-2, -2, -2, -2))
    
    # TODO: Can this be done better so we do not have to copy the actions for each player, but so that we can just use it as it is.
    
    p2_actions = np.tile(np.arange(muzero.actions), (curr_iset.shape[1], muzero.actions, 1))
    p1_actions = np.transpose(p2_actions, (0, 2, 1))
    next_p1_isets, next_p2_isets, next_utilities, next_terminal = vectorized_abstraction(curr_iset[0], curr_iset[1], p1_actions, p2_actions)
    
    # We will select only utilities of player 0. We can do some more fancy stuff here, but whatever.
    action_utility = next_utilities[..., 0] * legal
    # action_utility = legal * (next_utilities[..., 0] - next_utilities[..., 1]) / 2
    
    non_terminal = np.squeeze(validate_terminal(next_terminal), -1) * legal
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
    
    if depth + 1 == depth_limit:
      handle_mvs_layer(next_isets)
    else:
      handle_single_layer(next_isets, depth+1)
    
    
  def handle_gadget_layer(curr_iset, cf_values):
      
    iset_map, isets, actions = create_iset_map(curr_iset, 2) # Different amount of actions, only 2 for each player
    #iset_map, isets, actions = create_iset_map(curr_iset, max_actions)
    # TODO: Can these be done better?
    
    action_utilities = np.zeros((cf_values.shape[0], 2, 2))
    #action_utilities = np.zeros((cf_values.shape[0], max_actions, max_actions))
    # Resolving player plays the only legal action, while the other terminates the game
    action_utilities[:, 0, 0] = cf_values
    iset_legal = [np.ones(iset_map[pl].shape[:-1] + (2,)) for pl in range(2)]
    iset_legal[player][:, 1] = 0
    legals = np.ones((cf_values.shape[0], 2, 2))
    next_history = np.full((cf_values.shape[0], 2, 2), -1, dtype = int)
    if player == 0:
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
  init_condition = jnp.array([player == 0, player == 1])
  init_reaches = jnp.where(init_condition[..., None], init_reaches, 1) 
  
  constants = MuZeroCFRConstants(
    resolving_player = player,
    
    init_reaches = init_reaches,
    depth_actions = [a.shape[-1] for a in depth_history_actions],
    depth_iset_legal = convert_player_depth_to_jax(depth_iset_legal),
    
    depth_history_action_utility = convert_depth_to_jax(depth_history_action_utility),
    depth_history_iset = convert_depth_to_jax(depth_history_iset),
    depth_history_actions = convert_depth_to_jax(depth_history_actions),
    depth_history_legal = convert_depth_to_jax(depth_history_legal),
    
    depth_history_next_history = convert_depth_to_jax(depth_history_next_history),
  )
  depth_iset_map = convert_player_depth_to_jax(depth_iset_map)
  
  return MuZeroCFR(constants, depth_iset_map)
    


# The main idea is:
# Create root
class MuZeroGameplay:
  def __init__(self, muzero: MuZeroTrain, config: MuZeroGameplayConfig) -> None:
    self.config = config
    self.muzero = muzero
    self.mvs_actions = self.muzero.config.transformations + 1
    self.actions = muzero.actions
    self.new_game = True # flag that specifies whether we are at the beginning of the game or whether we have moved
    self.constructed_gadget = False
    self.initialize_isets() 
    
    self.cfr = None
    self.tree_depth = 0
    self.policy = {}
    
  # First finds the information states and public states from the game, then pushes them through abstraction layer
  def initialize_isets(self):
    if isinstance(self.muzero.game, JaxGoofspiel):
      key = jax.random.key(0)
      game_state, legals = self.muzero.game.initialize_structures(key) 
      _, *self.init_info = self.muzero.game.get_info(game_state) 
    else:
      state = self.muzero.game.new_initial_state()
      self.init_info =  np.array(state.information_state_tensor(0)), np.array(state.information_state_tensor(1)), np.array(state.public_state_tensor())
    p1_iset, p2_iset = self.muzero.get_both_abstraction(self.init_info[2], self.init_info[0], self.init_info[1])
    self.init_iset = np.stack((p1_iset[None, ...], p2_iset[None, ...]), 0) # Shape would be [Pl, H(D), 32]
    
    
  def reset(self):
    self.new_game = True 
     
  def build_initial_root(self, public_state, iset):
    assert np.allclose(iset, self.init_iset[self.config.player])
    assert np.allclose(public_state, self.init_info[2])
    reaches = np.ones((2,self.init_iset.shape[1]))
    cf_values = np.zeros((self.init_iset.shape[1],))
    return self.init_iset, reaches, cf_values

   
   
  def find_next_root(self, public_state, iset):
    return find_next_root(self.cfr, self.tree_depth, self.config.player, public_state, iset)

  
  def find_root_from_previous(self, public_state, iset):
    
  # We are passing public state and infoset separately, but from iset you should be able to get public state ideally.
    #interested in the reaches for the resolving player in the last layer
    last_layer_CF_vals = self.cfr.get_last_depth_player_cf_values(self.config.player)
    #abstracted_iset = self.muzero.get_abstraction(public_state, iset, self.config.player)
    reaches = self.cfr.find_reaches_from_average()
    #interested in the reaches for the resolving player in the last layer
    #[H(D)]
    last_layer_reaches = reaches[-1]
    #[Pl,H(D)]
    last_layer_iset_indices = jnp.stack([self.cfr.constants.depth_history_iset[-1][pl] for pl in range(2)], axis=0)
    #Assuming equal number of isets for both player
    #num_isets = last_layer_iset_indices[self.config.player].shape[0]
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
    found_node_indices = pub_state_mask.nonzero()[0]
    stacked_nodes = jnp.stack([last_layer_isets[0][found_node_indices], last_layer_isets[1][found_node_indices]], axis = 0)
    #Have to return reaches for both players
    return stacked_nodes, last_layer_reaches[:, found_node_indices], last_layer_CF_vals[found_node_indices]
    #This is a version without using the public state decoder
   
  
  # Starts in a single public state and creates a DL-tree.
  # Each layer should be done at once. Any call to NN should be done once!
  def prepare_cfr_structure(self, isets, reaches, cf_values, construct_gadget):
    self.cfr = prepare_cfr_structure(self.muzero, self.config.player, self.config.depth_limit, isets, reaches, cf_values, construct_gadget)

  def run_cfr(self):
    self.cfr.multiple_steps(self.config.resolve_iterations)

  def get_policy_from_cfr(self, iset):
    depth_limit = self.config.depth_limit + self.constructed_gadget
    if self.cfr is None or self.tree_depth >= depth_limit:
      return None
    policy = self.cfr.get_strategy(iset, self.config.player, self.tree_depth)
    policy = np.asarray(policy, dtype="float64")
    policy /= np.sum(policy)
    
    return policy
  
  def get_policy(self, public_state, iset):
     
    abstracted_iset = self.muzero.get_abstraction(public_state, iset, self.config.player)
    self.tree_depth += 1
    optional_policy = self.get_policy_from_cfr(abstracted_iset)
    if optional_policy is not None:
      return optional_policy
    
    construct_gadget = not self.new_game
    if self.new_game:
      isets, reaches, cf_values = self.build_initial_root(public_state, abstracted_iset)
      
      self.new_game = False
      self.constructed_gadget = False
      self.tree_depth = 0
      
    else:
      isets, reaches, cf_values = self.find_next_root(public_state, abstracted_iset)
      # With gadget the depth is one further, because of the opponents decision node.
      self.constructed_gadget = True
      self.tree_depth = 1

    self.prepare_cfr_structure(isets, reaches, cf_values, construct_gadget)  
    self.run_cfr()
    
    policy = self.get_policy_from_cfr(abstracted_iset)
    print("Policy: ", policy)
    
    return policy
  
  def get_action(self, public_state, iset):
    policy = self.get_policy(public_state, iset)
    return np.random.choice(self.actions, p=policy)
  