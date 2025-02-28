
from open_spiel.python.algorithms.mu_zero.jax_games.jax_game import JaxGame, JaxPolicy
from open_spiel.python.algorithms.mu_zero.jax_games.jax_goofspiel import JaxGoofspiel
from open_spiel.python.algorithms.mu_zero.experiments.utils import stringify, exploitability_from_spiel_jax_game
from open_spiel.python.algorithms.mu_zero.jax_games.jax_game_algorithms import exploitability_jax_game, nash_equilibrium_jax_game, create_random_policy


import pyspiel
import math
import numpy as np
   
  
def test_br(cards: int): 
  points_order = "descending" 
  
  game = JaxGoofspiel(cards=cards, points_order=points_order) 
  spiel_game = pyspiel.load_game("goofspiel", {"num_cards": cards, "points_order":  points_order, "imp_info": True})
  
  
  for seed in range(5):
    jax_policy = create_random_policy(game, seed)
    _, _, spiel_p1_exp, spiel_p2_exp = exploitability_from_spiel_jax_game(jax_policy, spiel_game, game) 
    p1_br, p2_br, jax_p1_exp, jax_p2_exp = exploitability_jax_game(game, jax_policy)
    print(spiel_p1_exp, jax_p1_exp)
    print(spiel_p2_exp, jax_p2_exp)
    assert math.isclose(jax_p1_exp, spiel_p1_exp, rel_tol=1e-3)
    assert math.isclose(jax_p2_exp, spiel_p2_exp, rel_tol=1e-3) 
  
  
def test_nash(cards: int):
    
  points_order = "descending" 
  
  game = JaxGoofspiel(cards=cards, points_order=points_order) 
  spiel_game = pyspiel.load_game("goofspiel", {"num_cards": cards, "points_order":  points_order, "imp_info": True})
  
  from pyinstrument import Profiler
  
  # profiler = Profiler()
  # profiler.start()
  cfr, nash_policy, nash_value = nash_equilibrium_jax_game(game)
  p1_br, p2_br, jax_p1_exp, jax_p2_exp = exploitability_jax_game(game, nash_policy)
  assert abs(jax_p1_exp - nash_value) < 1e-4
  assert abs(jax_p2_exp - nash_value) < 1e-4
  # profiler.stop()
  # print(profiler.output_text(unicode=True, color=True))
  
if __name__ == "__main__":
  # test_nash(3)
  # test_nash(4)
  # test_nash(5)
  # test_br(3)
  # test_br(4)
  test_br(5)