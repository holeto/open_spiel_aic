import jax
import jax.numpy as jnp
import numpy as np
from typing import Set

from open_spiel.python.algorithms.mu_zero.jax_games.jax_battleships import JaxBattleships
from open_spiel.python.algorithms.mu_zero.jax_games.jax_game_algorithms import nash_equilibrium_jax_game, exploitability_jax_game

def test_state_tensor_uniqueness(board_shape: tuple[int, int], ship_sizes: list[int]):
  """Test that state tensors are unique for each game history."""
  
  max_game_length = board_shape[0] * board_shape[1] + len(ship_sizes)
  game = JaxBattleships(board_shape=board_shape, ship_sizes=ship_sizes)
  key = jax.random.PRNGKey(0)
  state, legal_actions = game.initialize_structures(key)
  
  # Set to store state tensors
  state_tensors: Set[str] = set()
  
  def traverse_game_tree(current_state, current_legal_actions, depth=0):
    if depth > max_game_length:  # Prevent infinite recursion
      assert False, "Depth limit reached, Game should be over by now"
        
    # Get state tensor and convert to string for hashing
    state_tensor, p1_iset, p2_iset, public_state = game.get_info(current_state)
    state_tensor = np.array(state_tensor)
    state_tensor_str = str(state_tensor)
    
    # Verify state tensor is unique
    assert state_tensor_str not in state_tensors, f"Duplicate state tensor found at depth {depth}"
    state_tensors.add(state_tensor_str)
    
    # For each legal action, traverse the game tree
    for a1i, a1 in enumerate(current_legal_actions[0]):
      if a1 < 0.5:
        continue
      
      for a2i, a2 in enumerate(current_legal_actions[1]):
        if a2 < 0.5:
          continue
        
        new_state, is_terminal, reward, new_legal_actions = game.apply_action(
          current_state, key, depth, jnp.array([a1i, a2i]))
        if not is_terminal:
          traverse_game_tree(new_state, new_legal_actions, depth + 1)
  
  traverse_game_tree(state, legal_actions)
  print(f"Found {len(state_tensors)} unique state tensors")

def test_information_set_consistency(board_shape: tuple[int, int], ship_sizes: list[int]):
  """Test that information sets and public states are consistent with the game state."""
   
  max_game_length = board_shape[0] * board_shape[1] + len(ship_sizes)
  game = JaxBattleships(board_shape=board_shape, ship_sizes=ship_sizes)
  key = jax.random.PRNGKey(0)
  state, legal_actions = game.initialize_structures(key)
  
  p1_iset_map = {}
  p2_iset_map = {}
  public_info_map = {}
  p1_inverse_mapping = {}
  p2_inverse_mapping = {}
  public_inverse_mapping = {}
  
  def verify_information_sets(current_state, current_legal_actions, public_info = "", p1_info = "", p2_info = "", depth=0):
    if depth > max_game_length:  # Prevent infinite recursion
      assert False, "Depth limit reached, Game should be over by now"
    
  
    state_tensor, p1_iset, p2_iset, public_state = game.get_info(current_state)
     
    if p1_info in p1_iset_map: 
      assert jnp.allclose(p1_iset_map[p1_info], p1_iset), "Player 1's information set doesn't match"
    else:
      p1_iset_map[p1_info] = p1_iset
    
    if p2_info in p2_iset_map: 
      assert jnp.allclose(p2_iset_map[p2_info], p2_iset), "Player 2's information set doesn't match"
    else:
      p2_iset_map[p2_info] = p2_iset 
      
    if public_info in public_info_map:
      assert jnp.allclose(public_info_map[public_info], public_state), "Public state doesn't match"
    else:
      public_info_map[public_info] = public_state
    
    p1_iset = np.array(p1_iset)
    p2_iset = np.array(p2_iset)
    public_state = np.array(public_state)
    
    p1_iset_str = str(p1_iset)
    p2_iset_str = str(p2_iset)
    public_state_str = str(public_state)
    
    if p1_iset_str in p1_inverse_mapping:
      # print(current_state.shots)
      # print(p1_inverse_mapping[p1_iset_str])
      # print(p1_info)
      assert p1_inverse_mapping[p1_iset_str] == p1_info, "Player 1's information set doesn't match"
    else:
      p1_inverse_mapping[p1_iset_str] = p1_info
    
    if p2_iset_str in p2_inverse_mapping:
      assert p2_inverse_mapping[p2_iset_str] == p2_info, "Player 2's information set doesn't match"
    else:
      p2_inverse_mapping[p2_iset_str] = p2_info
    
    if public_state_str in public_inverse_mapping: 
      assert public_inverse_mapping[public_state_str] == public_info, "Public state doesn't match"
    else:
      public_inverse_mapping[public_state_str] = public_info
    
    # For each legal action, traverse the game tree
    for a1i, a1 in enumerate(current_legal_actions[0]):
      if a1 < 0.5:
        continue
      
      a1x, a1y = a1i % board_shape[1], a1i // board_shape[1]
      for a2i, a2 in enumerate(current_legal_actions[1]):
        if a2 < 0.5:
          continue
        
        a2x, a2y = a2i % board_shape[1], a2i // board_shape[1]
        
        if depth < len(ship_sizes):
          new_p1_info = p1_info + "Place:" + str(a1i) + "|"
          new_p2_info = p2_info + "Place:" + str(a2i) + "|"
          new_public_info = public_info + "Place|"
        else:
          p1_action_result = "P1 Shoot:" + str(a1i) + "-"
          p2_action_result = "P2 Shoot:" + str(a2i) + "-"
          if current_state.ships[1, a1y, a1x] > 0:
            hitted_ship = current_state.ships[1, a1y, a1x] - 1
            
            if current_state.ship_hits[0,hitted_ship] + 1 == current_state.ship_sizes[hitted_ship]:
              p1_action_result += "Sink" + str(hitted_ship)
            else:
              p1_action_result += "Hit"
          else:
            p1_action_result += "Miss"
          if current_state.ships[0, a2y, a2x] > 0:
            hitted_ship = current_state.ships[0, a2y, a2x] - 1 
            if current_state.ship_hits[1,hitted_ship] + 1 == current_state.ship_sizes[hitted_ship]:
              p2_action_result += "Sink" + str(hitted_ship)
            else:
              p2_action_result += "Hit"
          else:
            p2_action_result += "Miss"
            
          new_p1_info = p1_info + p1_action_result + "|" + p2_action_result + "|"
          new_p2_info = p2_info + p2_action_result + "|" + p1_action_result + "|"
          new_public_info = public_info + p1_action_result + "|" + p2_action_result + "|"
        
        new_state, is_terminal, reward, new_legal_actions = game.apply_action(
          current_state, key, depth, jnp.array([a1i, a2i]))
        
        
        
        if not is_terminal:
          verify_information_sets(new_state, new_legal_actions, new_public_info, new_p1_info, new_p2_info, depth + 1)
  
  verify_information_sets(state, legal_actions)
  print("Found", len(p1_iset_map), "unique player 1 information sets")
  print("Found", len(p2_iset_map), "unique player 2 information sets")
  print("Found", len(public_info_map), "unique public information sets")

def test_nash_equilibrium(board_shape: tuple[int, int], ship_sizes: list[int]): 
  game = JaxBattleships(board_shape=board_shape, ship_sizes=ship_sizes)
  cfr, nash_policy, nash_value = nash_equilibrium_jax_game(game)
  print(nash_policy.policy)
  # p1_br, p2_br, jax_p1_exp, jax_p2_exp = exploitability_jax_game(game, nash_policy) 

def verify_shapes():
  
  for game_settings in [((2, 2), [2]), ((3, 3), [3, 2]), ((5, 5), [2, 3, 4]), ((5, 3), [3, 3])]:
    print(*game_settings)
    game = JaxBattleships(*game_settings)
    
    init_state, legals = game.initialize_structures(jax.random.key(0))
    state, p1_iset, p2_iset, public_state = game.get_info(init_state)

    print(p1_iset.shape[0], game.information_state_tensor_shape())
    assert p1_iset.shape[0] == game.information_state_tensor_shape()
    assert p2_iset.shape[0] == game.information_state_tensor_shape()
    assert public_state.shape[0] == game.public_state_tensor_shape()
    

def print_policy():
  from open_spiel.python.algorithms.mu_zero.jax_games.jax_game_algorithms import stringify
  
  game = JaxBattleships(board_shape=(2, 2), ship_sizes=[2])
  cfr, nash_policy, nash_value = nash_equilibrium_jax_game(game)
  
  key = jax.random.key(0)
  def _traverse_tree(current_state, current_legal_actions, key, depth=0): 
    state_tensor, p1_iset, p2_iset, public_state = game.get_info(current_state)
    
    np_p1_iset = np.array(p1_iset)
    np_p2_iset = np.array(p2_iset)
    np_public_state = np.array(public_state)
    # print("SHIPS")
    # print(current_state.ships)
    # print("SHOTS")
    # print(current_state.shots) 
    print(stringify(np_public_state))
    print(stringify(np_p1_iset))
    print(stringify(np_p2_iset))
    print(nash_policy.policy[stringify(np_p1_iset)])
    print(nash_policy.policy[stringify(np_p2_iset)]) 
    for a1i, a1 in enumerate(current_legal_actions[0]):
      if a1 < 0.5:
        continue
      
      for a2i, a2 in enumerate(current_legal_actions[1]):
        if a2 < 0.5:
          continue
        
        action_key, next_key = jax.random.split(key)
        new_state, is_terminal, reward, new_legal_actions = game.apply_action(
          current_state, action_key, depth, jnp.array([a1i, a2i]))
        
        assert abs(reward) < 1.01
      
        if not is_terminal:
          assert abs(reward) < 0.01
          _traverse_tree(new_state, new_legal_actions, next_key, depth + 1)
    
    
  init_key, action_key = jax.random.split(key)
  init_state, legal_actions = game.initialize_structures(init_key) 
  _traverse_tree(init_state, legal_actions, action_key)
    
def check_abstraction():
  
  from open_spiel.python.algorithms.mu_zero.experiments.utils import load_model
  from open_spiel.python.algorithms.mu_zero.jax_games.jax_game_algorithms import stringify
  default_model_path = "muzero_networks/battleships_2x2_2/seed_43/muzero_"
  
  game = JaxBattleships(board_shape=(2, 2), ship_sizes=[2])
  key = jax.random.key(0)
  state, legal_actions = game.initialize_structures(key)
    
  state, is_terminal, reward, legal_actions = game.apply_action(state, key, 0, jnp.array([8, 8]))
  
  print(legal_actions)
  _, p1_iset, p2_iset, ps = game.get_info(state) 
  
  np_p1_iset = np.array(p1_iset)
  np_p2_iset = np.array(p2_iset)
  np_ps = np.array(ps)
  
  p1_iset_str = stringify(np_p1_iset)
  p2_iset_str = stringify(np_p2_iset)
  ps_str = stringify(np_ps)
  
  legals_leveled = 2 * legal_actions - 1
  
  for i in range(20, 21):
    model_path = default_model_path + str(i) + ".pkl"
    model = load_model(model_path)
    
    p1_abs, p2_abs, p1_probs, p2_probs, p1_sims, p2_sims = model.get_both_similarities_and_probs(ps, p1_iset, p2_iset)
    
    
    print(p1_probs)
    print(p1_sims)
    print(p2_probs)
    print(p2_sims)
    # p1_t = np.sum(np.abs(legals_leveled[0] - p1_sims), 1)
    # print(p1_t)
    # p2_t = np.sum(np.abs(legals_leveled[1] - p2_sims), 1)
    # print(p2_t)
    
    # print(p1_probs) 
    # print(p1_sims)
    
    
  
  
  

if __name__ == "__main__":
  board_shape = (2, 2)
  ship_sizes = [2]
  # verify_shapes()
  # test_state_tensor_uniqueness(board_shape, ship_sizes)
  # test_information_set_consistency(board_shape, ship_sizes)
  # test_nash_equilibrium(board_shape, ship_sizes)
  print_policy()