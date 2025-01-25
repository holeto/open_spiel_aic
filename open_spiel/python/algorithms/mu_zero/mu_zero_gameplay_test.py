
from open_spiel.python.algorithms.mu_zero.mu_zero import MuZeroTrain, MuZeroTrainConfig
from open_spiel.python.algorithms.mu_zero.mu_zero_gameplay import MuZeroGameplay, MuZeroGameplayConfig

from open_spiel.python.algorithms.mu_zero.jax_goofspiel import JaxOriginalGoofspiel
import numpy as np
import jax.numpy as jnp


def goofspiel_test(cards: int = 3, steps: int = 3, player=0):
  
  points_order = "descending"
  opp = 1 - player
  game = JaxOriginalGoofspiel(cards, points_order)
  
  config = MuZeroTrainConfig(batch_size=64, trajectory_max=cards-1, use_abstraction=True, entropy_schedule_size=(1000,), sampling_epsilon=0.8, abstraction_amount=10, transformations=4, similarity_metric="policy")
  
  muzero = MuZeroTrain(game, config)
  
  muzero.multiple_goofspiel_steps(10)
  
  gp_config =MuZeroGameplayConfig(player=player)
  muzero_gameplay = MuZeroGameplay(muzero, gp_config)
  
  init_info = game.initialize_structures()
  opp_legals = init_info[-1][opp]
  
  _, p1_iset, p2_iset, ps = game.get_info(*init_info[:-1])
  info = init_info[:-1]
  turn = 0
  
  for _ in range(steps):
    pl_action = muzero_gameplay.get_action(ps, p1_iset)
    #print(pl_action)
    #random opponent
    opp_action = np.random.choice(opp_legals)
    actions = [[],[]]
    actions[player] = pl_action
    actions[opp] = opp_action
    actions = jnp.stack(actions, axis=0)
    legals, rewards, point_cards, played_cards, p1_points = game.apply_action(*info, turn, actions)
    info = (point_cards, played_cards, p1_points)
    opp_legals = legals[opp]
    turn += 1
    _, p1_iset, p2_iset, ps = game.get_info(*info)

  

def main():
  cards = 3
  goofspiel_test(cards)
  

if __name__ == "__main__":
  main()