import pickle
import pyspiel
import jax
import numpy as np

from open_spiel.python.policy import TabularPolicy
from open_spiel.python.algorithms.best_response import BestResponsePolicy
from open_spiel.python.algorithms.mu_zero.jax_games.jax_game import JaxGame, JaxPolicy

def load_model(filepath: str):
  with open(filepath, "rb") as f:
    data= pickle.load(f)
  return data

def save_model(filepath: str, data):
   with open(filepath, "wb") as f:
    pickle.dump(data, f)

def stringify(a: list[float]) -> str:
  return ",".join(str(i) for i in a)



#IMPORTANT!!! Will only work for games without chance nodes.
def get_tabular_from_string(policy: JaxPolicy, spiel_game: pyspiel.Game, jax_game: JaxGame) -> TabularPolicy:
  tab_policy = TabularPolicy(spiel_game)
  def _traverse_tree(state, game_state, key, depth=0):
    if state.is_terminal():
      return
    p1_iset = state.information_state_string(0) 
    p2_iset = state.information_state_string(1)
    _, jax_p1, jax_p2, jax_ps = jax_game.get_info(game_state)
    jax_p1 = np.array(jax_p1)
    jax_p2 = np.array(jax_p2)


    for iset, jax_iset in zip([p1_iset, p2_iset], [jax_p1, jax_p2]):
      pol = tab_policy.policy_for_key(iset)
      for i in range(len(pol)):
        pol[i] = policy[stringify(jax_iset)][i]
    for a1 in state.legal_actions(0):
      for a2 in state.legal_actions(1):
        new_state = state.clone()
        new_state.apply_actions([a1, a2])
        next_key, action_key = jax.random.split(key)
        new_game_state, new_terminal, new_rewards, new_legals = jax_game.apply_action(game_state, action_key, depth, np.array([a1, a2]))
        _traverse_tree(new_state, new_game_state, next_key, depth +1)
  #using seed 0 here, as this is not a game with chance nodes it can be arbitrary
  start_key = jax.random.key(0)
  start_key, action_key = jax.random.split(start_key)
  _traverse_tree(spiel_game.new_initial_state(), *jax_game.initialize_structures(start_key)[:-1], action_key)
  return tab_policy    

 
def exploitability_from_spiel_jax_game(policy: JaxPolicy, spiel_game: pyspiel.Game, jax_game: JaxGame) -> tuple[float, float]:
    
  tab_policy = get_tabular_from_string(policy, spiel_game, jax_game)
  
  br1 = BestResponsePolicy(spiel_game, 1, tab_policy)
  br2 = BestResponsePolicy(spiel_game, 0, tab_policy)
   
  return  br1, br2, br1.value(spiel_game.new_initial_state()), br2.value(spiel_game.new_initial_state())