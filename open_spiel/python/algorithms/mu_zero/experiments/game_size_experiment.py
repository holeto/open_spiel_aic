import numpy as np

import jax

from open_spiel.python.algorithms.mu_zero.mu_zero_train import MuZeroTrain
from open_spiel.python.algorithms.mu_zero.mu_zero_gameplay import prepare_cfr_structure
from open_spiel.python.algorithms.mu_zero.experiments.utils import load_model



def get_abstracted_game_size( model: MuZeroTrain): 
  game = model.game
  init_reaches = np.ones((2, 1))
  init_cf_values = np.zeros((1, ))
  
  #can be arbitrary seed, as this game does not have chance nodes
  key = jax.random.key(0)
  state_key, init_key = jax.random.split(key)
  game_state, legals = game.initialize_structures(init_key) 
  _, init_p1_isets, init_p2_iset, init_ps = game.get_info(game_state) 
  
  p1_abstracted, p2_abstracted = model.get_both_abstraction(init_ps, init_p1_isets, init_p2_iset)

  init_iset = np.expand_dims(np.stack([p1_abstracted, p2_abstracted], axis=0), 1) 
  
  cfr = prepare_cfr_structure(model, 0, model.config.trajectory_max, init_iset, init_reaches, init_cf_values, False) 
  history_size = 0
  iset_size = 0
  for i in range(len(cfr.constants.depth_history_iset)):
    history_size += cfr.constants.depth_history_iset[i].shape[1]
    iset_size += cfr.depth_iset_map[i][0].shape[0] + cfr.depth_iset_map[i][1].shape[0]
  return history_size, iset_size


def get_abstracted_game_sizes(cards: int):
  if cards == 5:
    folder = "muzero_networks/train_seeds/goofspiel_5_descending"
    ks = [5, 10, 15, 20, 25, 30, 35]
    iters = 100
  elif cards == 4:
    folder = "muzero_networks/train_seeds/goofspiel_4_descending"
    ks = [1, 2, 3, 4, 5, 6, 7]
    iters = 80
  similarity_metric = ["legal_actions", "policy", "legal_policy", "action_history", "action_history_legal_policy"] 
  amount_seeds = 10
  
  for k in ks:
    for sim_id, similarity in enumerate(similarity_metric):
      if "policy" not in similarity:
        print(f"We are skipping {similarity} similarity metric")
        continue
      for seed in range(amount_seeds):
        final_seed = seed + k * 10 + sim_id * 1000
        model_path = f"{folder}/seed_{final_seed}/muzero_{iters}.pkl"
        model = load_model(model_path)
        history_size, iset_size = get_abstracted_game_size(model)
        print(f"Seed: {final_seed}|History size: {history_size}|Iset size: {iset_size}")
        

if __name__ == "__main__":
  model = load_model("muzero_networks/goofspiel_5_descending/seed_90305/muzero_200.pkl")
  print(get_abstracted_game_size(model))