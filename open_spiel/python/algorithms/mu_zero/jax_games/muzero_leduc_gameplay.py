
import jax
import jax.numpy as jnp
import jax.lax as lax

import chex
import numpy as np


from open_spiel.python.algorithms.mu_zero.mu_zero_train import MuZeroTrain
from open_spiel.python.algorithms.mu_zero.jax_games.muzero_leduc_cfr import MuZeroLeducCFRConstants, MuZeroLeducCFR, check_iset_similarity
from open_spiel.python.algorithms.mu_zero.mu_zero_gameplay import MuZeroGameplayConfig
from open_spiel.python.algorithms.mu_zero.jax_games.jax_leduc import JaxLeduc, LeducGameState

def convert_player_depth_to_jax(arr):
  return [[jnp.array(p) for p in d] for d in arr]

def tree_where(pred: chex.Array, x: chex.ArrayTree, y: chex.ArrayTree) -> chex.ArrayTree:
  
  def _where(x, y):
    shape_difference = len(x.shape) - len(pred.shape)
    element_pred = pred
    for i in range(shape_difference):
      element_pred = element_pred[..., None]
    return jnp.where(element_pred, x, y)
  
  return jax.tree.map(_where, x, y)


def get_real_pure_mvs(after_chance_state: LeducGameState):
  """Returns a 12x12 matrix of true MVS values 
  in a given state at the root of the post-chance subgame.
  The MVS will always be returned with values for player one
  (just multiply by -1 for player 2) and has player
  1 as the row player and player 2 as the column player."""
  no_raise_win = after_chance_state.current_chips[1] / 13
  one_raise_win = (after_chance_state.current_chips[1] + 4) / 13
  no_raise_loss = - after_chance_state.current_chips[0] / 13
  one_raise_loss = - (after_chance_state.current_chips[0] + 4) / 13
  private_card_bins = jnp.floor_divide(after_chance_state.private_cards, 2)
  public_card_matched = private_card_bins == jnp.floor_divide(after_chance_state.public_card - 1, 2) 
  player_won = jnp.logical_or(public_card_matched[0], (jnp.logical_and(~public_card_matched[1], private_card_bins[0] > private_card_bins[1])))
  tie = jnp.logical_and(~player_won, private_card_bins[0] == private_card_bins[1])
  k1 = jnp.where(player_won, no_raise_win, no_raise_loss)
  k1 = jnp.where(tie, 0, k1)
  k2 = jnp.where(player_won, one_raise_win, one_raise_loss)
  k2 = jnp.where(tie, 0, k2)
  k3 = jnp.where(player_won, (after_chance_state.current_chips[1] + 8) / 13,  - (after_chance_state.current_chips[0] + 8) / 13)
  k3 = jnp.where(tie, 0, k3)
  mvs = [[k1] * 6 + [k2] * 6,
         [k1] * 6 + [k2] * 6,
         [k1] * 6 + [k3, one_raise_win] * 3 ,
         [k1] * 6 + [no_raise_loss] * 6,
         [k1] * 6 + [no_raise_loss] * 6,
         [k1] * 6 + [k3, one_raise_win] * 3 ,
         ([k2] * 2 + [no_raise_win] * 2 + [k3] * 2) * 2,
         ([k2] * 2 + [no_raise_win] * 2 + [one_raise_loss] * 2) * 2,
         ([k2] * 2 + [no_raise_win] * 2 + [one_raise_loss] * 2) * 2,
         ([k2] * 2 + [no_raise_win] * 2 + [one_raise_loss] * 2) * 2,
         ([k2] * 2 + [no_raise_win] * 2 + [k3] * 2) * 2,
         ([k2] * 2 + [no_raise_win] * 2 + [k3] * 2) * 2,]
  mvs = jnp.array(mvs)
  #jax.debug.breakpoint()
  return mvs

def expand_chance(game: JaxLeduc, after_chance_node_state: LeducGameState):
  chance_node_states, validity_mask = game.generate_pc_nodes_and_mask(after_chance_node_state)
  chance_node_states = jax.tree_map(lambda *x: jnp.stack(x), *chance_node_states)
  return chance_node_states, validity_mask

def tree_index(choices: chex.Array, x:chex.ArrayTree)->chex.ArrayTree:
  tree_leaves, tree_def = jax.tree_flatten(x)
  new_leaves = []
  for leaf in tree_leaves:
    leaf = np.asarray(leaf)
    new_leaf = leaf[choices]
    new_leaves.append(new_leaf)
  new_tree = jax.tree_unflatten(tree_def, new_leaves)
  return new_tree


def convert_depth_to_jax(arr):
  return [jnp.array(d) for d in arr]

def validate_terminal(terminal, threshold: float = 0.5):
  return terminal < threshold

def validate_chance(prev_game_state_pc, game_state_pc):
  return np.squeeze(np.logical_and(prev_game_state_pc == 0, game_state_pc > 0), axis=-1)



def find_next_root(cfr: MuZeroLeducCFR, tree_depth: int, player: int, public_state, iset, isets_to_states):
  opponent = 1 - player
  public_state_histories = cfr.find_public_state_from_iset(iset, player, tree_depth)
  history_reaches, history_chance_reaches = cfr.find_reaches_from_average()
  history_reaches, history_chance_reaches = history_reaches[tree_depth], history_chance_reaches[tree_depth]
  history_reaches, history_chance_reaches = history_reaches[:, public_state_histories], history_chance_reaches[public_state_histories]
  next_reaches = np.where(np.array([[player == 0], [player == 1]]), history_reaches * history_chance_reaches[None, ...], 1.0) #* 0.25
  depth_isets = np.array(cfr.constants.depth_history_iset[tree_depth])
  depth_cf_vals = np.array(cfr.cf_values[tree_depth][opponent])
  next_isets_id = depth_isets[:, public_state_histories]
  next_cf_values = depth_cf_vals[next_isets_id[opponent]]
  next_isets = cfr.depth_iset_map[tree_depth][opponent][next_isets_id[opponent]]
  next_isets = np.stack([np.array(cfr.depth_iset_map[tree_depth][pl])[next_isets_id[pl]] for pl in range(2)], axis = 0)
  next_states, next_legals = isets_to_states(next_isets[0], next_isets[1])
  return next_states, next_legals, next_reaches, next_cf_values 


# Starts in a single public state and creates a DL-tree.
# Each layer should be done at once. Any call to NN should be done once!
# For now this creates the tree in the original game and not an abstraction tree
def prepare_cfr_structure(muzero: MuZeroTrain, player: int, turns, depth_limit, states, legals, reaches, cf_values, construct_gadget):
  chex.assert_equal(states.terminal.shape[0], reaches.shape[1])
  chex.assert_equal(states.terminal.shape, cf_values.shape)
  game = muzero.game
  assert isinstance(game, JaxLeduc), """This is a domain specific implementation that works only for JaxLeduc!"""
  mvs_actions = 12 
  #mvs_actions = muzero.config.transformations + 1
  
  depth_iset_map = [] # We create initial dummy iset 0
  depth_iset_legal = []
  
  depth_history_action_utility = []
  depth_history_iset = []
  depth_history_actions = []
  depth_history_legal = []
  depth_history_is_chance = []
  
  depth_history_next_history = []

  vectorized_get_info = jax.vmap(game.get_info, in_axes=(0), out_axes=(0, 0, 0, 0))
  vectorized_expand_chance = jax.vmap(expand_chance, in_axes=(None, 0), out_axes=(0))
  #Key is not mapped, since we build a full tree
  # and do not care which chance nodes outcomes get sampled
  dummy_key = jax.random.key(0)
  #Just a wrapper method to be able
  #to vmap over turn without throwing errors
  def next_turn_wrapper(state, key, turn, joint_action):
    turn = turn[0]
    return game.apply_action(state, key, turn, joint_action)

  vectorized_next_state = jax.vmap(next_turn_wrapper, in_axes=(0, None, 0, 1), out_axes=(0, 0, 0, 1))
  vectorized_get_mvs = jax.vmap(get_real_pure_mvs, in_axes=(0), out_axes=(0))
  
  not_acting_legals = np.array([1, 0, 0, 0])
  chance_legals = np.array([[1, 0, 0, 0], [1, 1, 1, 1]])

  # TODO: Split the map to be separate for each depth.
  # Because of imperfect recall it does not make sense to have all the isets in the same map.
  def create_iset_map(curr_iset, amount_actions, curr_legal=None):
    isets = [[], []]
    iset_map = [[], []]
    iset_legal = [[], []]
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
          if curr_legal is not None:
            iset_legal[pl].append(curr_legal[pl, i])
        isets[pl].append(curr_index)
        
    isets = np.array(isets)
    actions = isets[..., None] * amount_actions + np.arange(amount_actions)[None, None, ...] 
    iset_map = [np.array(i) for i in iset_map]
    iset_legal = [np.array(i) for i in iset_legal]
    return iset_map, iset_legal, isets, actions
  
  
  def handle_mvs_layer(curr_states):
    depth_history_is_chance.append(np.zeros(curr_states.terminal.shape[0], dtype=bool))
    state_tensors, p1_isets, p2_isets, public_states = vectorized_get_info(curr_states)
    curr_iset = np.stack((p1_isets, p2_isets))
    iset_map, _, isets, actions = create_iset_map(curr_iset, mvs_actions)
    #mvs_vals = muzero.get_mvs(public_states, p1_isets, p2_isets)
    mvs_vals = vectorized_get_mvs(curr_states)
    iset_legal = [np.ones(iset_map[pl].shape[:-1] + (mvs_actions,)) for pl in range(2)]  
    legal = np.ones_like(mvs_vals)
    next_history = np.full_like(mvs_vals, -1, dtype=int)
    
    depth_iset_map.append(iset_map)
    depth_iset_legal.append(iset_legal)
    depth_history_action_utility.append(mvs_vals)
    depth_history_iset.append(isets)
    depth_history_actions.append(actions) 
    depth_history_legal.append(legal) 
    depth_history_next_history.append(next_history)
    
  
  def handle_single_layer(curr_turns, curr_states, curr_legal, valid, is_chance, depth):
    state_tensors, p1_isets, p2_isets, public_states = vectorized_get_info(curr_states)
    invalid_iset = np.zeros_like(p1_isets[0])
    curr_iset = np.stack((p1_isets, p2_isets))
    curr_iset = np.where(is_chance[None, :, None], invalid_iset[None, None, ...], curr_iset)
    iset_map, iset_legal, isets, actions = create_iset_map(curr_iset, muzero.actions, curr_legal = curr_legal)
    #breakpoint()
    p1_legal_iset, p2_legal_iset = iset_legal[0], iset_legal[1]
    p1_legal_iset, p2_legal_iset = p1_legal_iset > 0, p2_legal_iset > 0
    
    p1_legal, p2_legal = curr_legal[0], curr_legal[1]
    legal = p1_legal[..., None] * p2_legal[..., None, :]
    # If we ever change to Bool[D, H(D),Pl, A], Instead of [D, H(D),A1, A2]
    # legal_stacked = np.stack((p1_legal, p2_legal), 0)
    
    
    # Even with in dimension -1, we want output dimension to be before the last dimension.
    # Checked that this does what is supposed (which is np.transpose(res, (0, 2, 3, 1)))
    #vectorized_abstraction = jax.vmap(jax.vmap(muzero.get_next_state_from_abstraction, in_axes=(None, None, -1, -1), out_axes=(-2, -2, -2, -2)), in_axes=(None, None, -1, -1), out_axes=(-2, -2, -2, -2))
    
    p1_actions = np.tile(np.repeat(np.arange(muzero.actions), muzero.actions), curr_iset.shape[1])
    p2_actions = np.tile(np.tile(np.arange(muzero.actions), muzero.actions), curr_iset.shape[1])
    joint_actions = np.stack((p1_actions, p2_actions))
    prev_states = jax.tree_map(lambda x: jnp.repeat(x, muzero.actions ** 2, axis=0), curr_states)
    valid = np.repeat(valid, muzero.actions ** 2, axis=0)
    curr_turns = np.repeat(curr_turns, muzero.actions ** 2, axis=0)

    next_states, next_terminal, next_utilities, next_legals = vectorized_next_state(prev_states, dummy_key, curr_turns, joint_actions)

    # We will select only utilities of player 0. We can do some more fancy stuff here, but whatever.
    #Hopefully this should keep that player zero is the row player
    
    next_chance =  validate_chance(prev_states.public_card, next_states.public_card)
    depth_history_is_chance.append(is_chance)
    is_chance = np.repeat(is_chance, muzero.actions ** 2, axis=0).astype(bool)

    #Here we handle chance nodes.
    #expand them and then replace the action
    # outcomes with chance node outcomes for them
    chance_outcome_states, chance_valid = vectorized_expand_chance(muzero.game, curr_states)
    #now properly reshape into H(D) like shape
    after_chance_shape = (game.total_cards) * curr_states.terminal.shape[0]
    chance_outcome_states = jax.tree_map(lambda x: x.reshape((after_chance_shape, ) + x.shape[2:]), chance_outcome_states)
    chance_outcome_states = tree_index(np.asarray(chance_valid, bool).ravel(), chance_outcome_states)
    #now we just need to pad the chance outcomes
    # to be as if 4 ** 2 actions were played and not just 4
    chance_outcome_states = jax.tree_map(lambda x: x.reshape((-1, 4) + x.shape[1:]), chance_outcome_states)
    chance_outcome_states = jax.tree_map(lambda x: jnp.repeat(x, 4, axis=0), chance_outcome_states)
    chance_outcome_states = jax.tree_map(lambda x: x.reshape((-1, ) + x.shape[2:]), chance_outcome_states)

    #Replace chance outcomes
    next_states = tree_where(is_chance, chance_outcome_states, next_states)

    next_valid = ~is_chance * valid
    next_utilities = np.where(next_valid, next_utilities, 0)
    next_terminal = np.where(next_valid, next_terminal, False)

    next_turns = curr_turns + 1
    
    next_utilities = next_utilities.reshape(legal.shape)
    next_terminal = next_terminal.reshape(legal.shape)
    
    action_utility = next_utilities * legal

    #Need to give invalid copied states
    # only if this state is not valid, but give 
    # the invalid legals immediately!
    # This is done so that chance node 
    # outcomes are propagated to MVS layer and given
    # value there
    next_states = tree_where(valid, next_states, prev_states)
    #For chance nodes switch to chance legals
    next_legals = np.where(next_chance[None, :, None], chance_legals[:, None, ...], next_legals)
    #otherwise get only the single legal action there
    next_legals = np.where(next_valid[None, ..., None], next_legals, not_acting_legals[None, None, ...])
    non_terminal = validate_terminal(next_terminal) * legal
    
    
    # From [H(D), A1, A2] should select [H(D + 1)] 
    # nonzero() returns indices which are non zero in tuple (4-tuple in this case)
    nonzeros = non_terminal.flatten().nonzero()
    next_states = jax.tree_map(lambda x: x[*nonzeros], next_states)
    next_valid = next_valid[*nonzeros]
    next_legals = next_legals[:, *nonzeros, :]
    next_chance = next_chance[*nonzeros]
    next_turns = next_turns[*nonzeros]
    
    # This should be -1 everywhere, except the part where you have next history. Therey ou go by terminal and just add 1
    next_history = (np.cumsum(non_terminal).reshape(non_terminal.shape) * non_terminal) - 1


    depth_iset_map.append(iset_map)
    depth_iset_legal.append(iset_legal)
    depth_history_action_utility.append(action_utility)
    depth_history_iset.append(isets)
    depth_history_actions.append(actions)
    depth_history_legal.append(legal)  
    depth_history_next_history.append(next_history.astype(int))
    
    #breakpoint()
    if np.all(next_history < 0):
      return
    
    if depth + 1 == depth_limit:
      handle_mvs_layer(next_states)
    else:
      handle_single_layer(next_turns, next_states, next_legals, next_valid, next_chance, depth+1)
    
    
  def handle_gadget_layer(curr_states, curr_legal, cf_values):

    state_tensors, p1_isets, p2_isets, public_states = vectorized_get_info(curr_states)
    curr_iset = np.stack((p1_isets, p2_isets))  
    iset_map, _ ,isets, actions = create_iset_map(curr_iset, 2) # Different amount of actions, only 2 for each player
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
    depth_history_is_chance.append(np.zeros(cf_values.shape[0], dtype=bool))
    depth_iset_legal.append(iset_legal)
    depth_history_iset.append(isets)
    depth_history_actions.append(actions)
    depth_history_action_utility.append(action_utilities)
    depth_history_legal.append(legals)
    depth_history_next_history.append(next_history)
    
    handle_single_layer(turns, curr_states, curr_legal, np.ones(curr_states.terminal.shape[0]), np.zeros(curr_states.terminal.shape[0]), 0)   

  if construct_gadget:
    handle_gadget_layer(states, legals, cf_values)
  else:
    handle_single_layer(turns, states, legals, np.ones(states.terminal.shape[0]), np.zeros(states.terminal.shape[0]), 0)
  init_reaches = jnp.copy(reaches)
  init_condition = jnp.array([player == 0, player == 1])
  init_reaches = jnp.where(init_condition[..., None], init_reaches, 1) 
  
  constants = MuZeroLeducCFRConstants(
    resolving_player = player,
    
    init_reaches = init_reaches,
    depth_actions = [a.shape[-1] for a in depth_history_actions],
    depth_iset_legal = convert_player_depth_to_jax(depth_iset_legal),
    
    depth_history_action_utility = convert_depth_to_jax(depth_history_action_utility),
    depth_history_iset = convert_depth_to_jax(depth_history_iset),
    depth_history_actions = convert_depth_to_jax(depth_history_actions),
    depth_history_legal = convert_depth_to_jax(depth_history_legal),
    
    depth_history_next_history = convert_depth_to_jax(depth_history_next_history),
    depth_history_is_chance = convert_depth_to_jax(depth_history_is_chance)
  )
  depth_iset_map = convert_player_depth_to_jax(depth_iset_map)
  
  return MuZeroLeducCFR(constants, depth_iset_map)
    


# The main idea is:
# Create root
class MuZeroLeducGameplay:
  def __init__(self, muzero: MuZeroTrain, config: MuZeroGameplayConfig) -> None:
    self.config = config
    assert self.config.depth_limit == 5, "This gameplay version was designed with depth limit 5, which creates one CFR before the chance node and one after it."
    self.muzero = muzero
    assert isinstance(muzero.game, JaxLeduc), "This is a domain specific implementation for JaxLeduc!"
    self.mvs_actions = self.muzero.config.transformations + 1
    self.actions = muzero.actions
    self.new_game = True # flag that specifies whether we are at the beginning of the game or whether we have moved
    self.constructed_gadget = False
    self.isets_to_states = jax.vmap(muzero.game.reconstruct_state_from_isets, in_axes=(0), out_axes=(0, 1))
    self.initialize_states() 
    
    self.cfr = None
    self.tree_depth = 0
    self.round = 0
    self.cfr_start_turn = 0
    self.policy = {}
    
  # First finds the information states and public states from the game, then pushes them through abstraction layer
  def initialize_states(self):
    states, init_legals = self.muzero.game.generate_all_private_card_nodes() # Shape would be [H(D), ??]
    #This esentially creates one large
    # LeducGameState, which is a batch of the states
    #that can then be vmapped over
    self.init_states = jax.tree_map(lambda *x: jnp.stack(x), *states)
    self.init_legals = np.tile(init_legals[:, None, ...], (1, self.init_states.terminal.shape[0], 1))
    #breakpoint()
    
    
  def reset(self):
    self.cfr_start_turn = 0
    self.new_game = True 
     
  def build_initial_root(self, public_state, iset):
    #assert np.allclose(iset, self.init_iset[self.config.player])
    #assert np.allclose(public_state, self.init_info[2])
    reaches = np.full((2, self.init_states.terminal.shape[0]), 1 / 30)
    cf_values = np.zeros((self.init_states.terminal.shape[0]))
    return self.init_states, self.init_legals, reaches, cf_values

   
   
  def find_next_root(self, public_state, iset):
    return find_next_root(self.cfr, self.tree_depth, self.config.player, public_state, iset, self.isets_to_states)

   
  
  # Starts in a single public state and creates a DL-tree.
  # Each layer should be done at once. Any call to NN should be done once!
  def prepare_cfr_structure(self, turns, states, legals, reaches, cf_values, construct_gadget):
    self.cfr = prepare_cfr_structure(self.muzero, self.config.player, turns, self.config.depth_limit, states, legals, reaches, cf_values, construct_gadget)

  def run_cfr(self):
    self.cfr.multiple_steps(self.config.resolve_iterations)

  def get_policy_from_cfr(self, iset):
    #a bit of a hack how to get public card from the
    #iset
    public_card = np.nonzero(iset[self.muzero.game.total_cards + 2:2 * self.muzero.game.total_cards + 3])[0][0]
    #breakpoint()
    #Encountered an iset after chance node
    # but we are still in the first CFR
    if public_card > 0 and self.round == 0:
      print("After chance node encountered")
      return None 
    depth_limit = self.config.depth_limit + self.constructed_gadget
    if self.cfr is None or self.tree_depth >= depth_limit:
      return None
    policy = self.cfr.get_strategy(iset, self.config.player, self.tree_depth)
    policy = np.asarray(policy, dtype="float64")
    policy /= np.sum(policy)
    print("Policy: ", policy)
    
    return policy
  
  def get_policy(self, public_state, iset):
    
    self.tree_depth += 1
    optional_policy = self.get_policy_from_cfr(iset)
    if optional_policy is not None:
      
      return optional_policy
    
    construct_gadget = not self.new_game
    if self.new_game:
      states, legals, reaches, cf_values = self.build_initial_root(public_state, iset)
      
      self.new_game = False
      self.constructed_gadget = False
      self.tree_depth = 0
      
    else:
      self.cfr_start_turn = self.tree_depth
      self.round = 1
      print("Rebuilding CFR")
      print("Start turn: ", self.cfr_start_turn)
      #Always search for the next root at depth_limit
      self.tree_depth = self.config.depth_limit
      states, legals, reaches, cf_values = self.find_next_root(public_state, iset)
      # With gadget the depth is one further, because of the opponents decision node.
      self.constructed_gadget = True
      self.tree_depth = 1

    turns = np.full((states.terminal.shape[0], 1), self.cfr_start_turn, dtype=int)
    #jax.debug.breakpoint()
    self.prepare_cfr_structure(turns, states, legals, reaches, cf_values, construct_gadget)  
    self.run_cfr()
    
    policy = self.get_policy_from_cfr(iset)
    
    return policy

  def get_action(self, public_state, iset):
    policy = self.get_policy(public_state, iset)
    return np.random.choice(self.actions, p=policy)
  
  
  