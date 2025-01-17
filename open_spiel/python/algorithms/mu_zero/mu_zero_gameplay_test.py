
from open_spiel.python.algorithms.mu_zero.mu_zero import MuZeroTrain, MuZeroTrainConfig
from open_spiel.python.algorithms.mu_zero.mu_zero_gameplay import MuZeroGameplay, MuZeroGameplayConfig

from open_spiel.python.algorithms.mu_zero.jax_goofspiel import JaxOriginalGoofspiel


def goofspiel_test(cards: int = 3):
  
  points_order = "descending"
  game = JaxOriginalGoofspiel(cards, points_order)
  
  config = MuZeroTrainConfig(batch_size=64, trajectory_max=cards-1, use_abstraction=True, entropy_schedule_size=(1000,), sampling_epsilon=0.8, abstraction_amount=10, transformations=4, similarity_metric="policy")
  
  muzero = MuZeroTrain(game, config)
  
  muzero.multiple_goofspiel_steps(10)
  
  gp_config =MuZeroGameplayConfig(player=0)
  muzero_gameplay = MuZeroGameplay(muzero, gp_config)
  
  init_info = game.initialize_structures()
  
  _, p1_iset, p2_iset, ps = game.get_info(*init_info[:-1])
  
  muzero_gameplay.get_action(ps, p1_iset)
  

def main():
  cards = 3
  goofspiel_test(cards)
  

if __name__ == "__main__":
  main()