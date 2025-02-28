import scipy
import os
import numpy as np
from scipy.cluster.vq import kmeans, vq, whiten
import matplotlib.pyplot as plt
import jax
import pyspiel
import matplotlib.pyplot as plt
 
from open_spiel.python.algorithms.mu_zero.jax_games.jax_game import JaxGame, JaxPolicy
from open_spiel.python.algorithms.mu_zero.jax_games.jax_goofspiel import JaxGoofspiel, JaxModifiedGoofspiel
from open_spiel.python.algorithms.mu_zero.jax_games.jax_game_algorithms import nash_equilibrium_jax_game, exploitability_jax_game, prepare_cfr_from_game, extract_policy_from_cfr

from open_spiel.python.algorithms.mu_zero.experiments.utils import stringify


def perform_kmeans(data, k, normalize=True, plot_results=False, random_seed=None):
  """
  Perform K-means clustering on the input data.

  Args:
      data: numpy array of shape (n_samples, n_features)
      k: int, number of clusters
      normalize: bool, whether to normalize/whiten the data
      plot_results: bool, whether to plot the results (only works for 2D data)
      random_seed: int or None, seed for reproducibility

  Returns:
      centroids: numpy array of shape (k, n_features), the cluster centers
      labels: numpy array of shape (n_samples,), the assigned cluster for each data point
      distortion: float, the mean distortion
  """
  # Set random seed if provided
  if random_seed is not None:
    np.random.seed(random_seed)

  # Make a copy of the data to avoid modifying the original
  data_copy = np.array(data, dtype=np.float64)

  # Normalize/whiten the data if requested
  if normalize:
    data_copy = whiten(data_copy)

  # Perform K-means clustering
  centroids, distortion = kmeans(data_copy, k)

  # Assign each data point to the nearest centroid
  labels, distances = vq(data_copy, centroids)
  return centroids, labels, distortion

  # Plot the results if requested and data is 2D
  if plot_results and data_copy.shape[1] == 2:
    plt.figure(figsize=(10, 8))

    # Plot the data points, colored by cluster
    for i in range(k):
      cluster_points = data_copy[labels == i]
      plt.scatter(cluster_points[:, 0],
                  cluster_points[:, 1], label=f'Cluster {i}')

    # Plot the centroids
    plt.scatter(centroids[:, 0], centroids[:, 1], s=200,
                c='black', marker='X', label='Centroids')

    plt.title(f'K-means Clustering (k={k})')
    plt.legend()
    plt.show()

  return centroids, labels, distortion


def get_all_public_states_with_isets_and_similarites(game: JaxGame, policy_dict: dict) -> tuple[dict, dict]:

 
  # Initialize empty dictionary for storing state-infoset mappings
  state_infoset_map = {}
  state_similarity_map = {}

  # Initialize the game state
  key = jax.random.key(0)
  state_key, init_key = jax.random.split(key)
  game_state, legals = game.initialize_structures(init_key)

  # Get initial info 
  
  
  # print(model.get_both_similarities_and_probs(init_ps, init_p1_isets, init_p2_iset)[2:4])
  def _traverse_tree(game_state, legal_actions, key, depth=0):
    state, p1_iset_tensor, p2_iset_tensor, ps = game.get_info(game_state)
    ps = stringify(ps)
    p1_iset = stringify(p1_iset_tensor)
    p2_iset = stringify(p2_iset_tensor)
    
    if ps not in state_infoset_map:
      state_infoset_map[ps] = [[], []]
      state_similarity_map[ps] = [[], []]
      
    if p1_iset not in state_infoset_map[ps][0]:
      state_infoset_map[ps][0].append(p1_iset)
      state_similarity_map[ps][0].append(policy_dict[p1_iset])
      # state_similarity_map[ps][0].append(p1_iset_tensor)
      # state_similarity_map[ps][0].append(np.array(legal_actions[0]))
      
    if p2_iset not in state_infoset_map[ps][1]:
      state_infoset_map[ps][1].append(p2_iset)
      state_similarity_map[ps][1].append(policy_dict[p2_iset])
      # state_similarity_map[ps][1].append(p2_iset_tensor)
      # state_similarity_map[ps][1].append(np.array(legal_actions[1]))
    
    for a1i, a1 in enumerate(legal_actions[0]):
      if a1 < 0.5:
        continue
      for a2i, a2 in enumerate(legal_actions[1]):
        if a2 < 0.5:
          continue
        next_key, action_key = jax.random.split(key)
        new_game_state, new_terminal, new_rewards, new_legals = game.apply_action(
            game_state, action_key, depth, np.array([a1i, a2i]))

        if new_terminal:
          continue 
        
        _traverse_tree(new_game_state, new_legals, next_key, depth + 1)

  _traverse_tree(game_state, legals, state_key)
  
  return state_infoset_map, state_similarity_map


def concat_ps_sim(state, sim):
  return np.concatenate([state, sim])




def compute_nash_wtih_cluster(game, cluster_map):
  muzero = prepare_cfr_from_game(game, state_cluster_map) 
  muzero.multiple_steps(2000) 
  policy_dict = extract_policy_from_cfr(game, muzero, state_cluster_map) 
  return policy_dict
  

def main():
  
  cards = 4
  turns = 3
  first_card = 6
  points_order = "descending"
  # game = JaxModifiedGoofspiel(cards, turns, first_card)
  game = JaxGoofspiel(cards=cards, points_order=points_order) 
  spiel_game = pyspiel.load_game("goofspiel", {"num_cards": cards, "points_order":  points_order, "imp_info": True})
  spiel_game_tb = pyspiel.load_game_as_turn_based("goofspiel", {"num_cards": cards, "points_order": points_order, "imp_info": True})
   
  
  from open_spiel.python.jax.cfr.jax_cfr import JaxCFR
  
  # j_cfr = JaxCFR(spiel_game_tb)
  # j_cfr.multiple_steps(1000)
  # dict_nash = j_cfr.average_policy()
  
  # print(dict_nash.action_probabilities(spiel_game_tb.new_initial_state()))
  
  _, dict_nash, nash_value = nash_equilibrium_jax_game(game)
  
  init_state, init_p1_iset, init_p2_iset, init_ps = game.get_info(game.initialize_structures(jax.random.key(0))[0])
  
  init_state = np.array(init_state)
  init_p1_iset = np.array(init_p1_iset)
  init_p2_iset = np.array(init_p2_iset)
  
  # print(dict_nash[stringify(init_p1_iset)])
  # print(nash_value)
  
  state_iset_map, state_sim_map = get_all_public_states_with_isets_and_similarites(game, dict_nash)
    
  max_k = 10
  p1_exps, p2_exps= np.zeros(max_k), np.zeros(max_k)
  
  for k in range(1, max_k):
    state_cluster_map = {}
    # state_cluster_label_map = {}
    for state, sims in state_sim_map.items():
      state_cluster_map[state] = [{}, {}]
      # state_cluster_label_map[state] = [{}, {}]
      for pl in range(2):
        pl_sims = np.array(sims[1]) 
        cluster_amount = min(k, pl_sims.shape[0])
        center, labels, distance = perform_kmeans(pl_sims, cluster_amount, normalize=False, random_seed=7)
        for iset_id, iset in enumerate(state_iset_map[state][pl]):
          state_cluster_map[iset] = center[labels[iset_id]]
          # state_cluster_map[state][pl][iset] = center[labels[iset_id]]
          # state_cluster_label_map[state][pl][iset] = labels[iset_id] 
    
    policy_dict = compute_nash_wtih_cluster(game, state_cluster_map)
    _, _, p1_exp, p2_exp = exploitability_jax_game(game, policy_dict) 
    
    
    p1_exps[k] = p1_exp
    p2_exps[k] = p2_exp

  xs = np.arange(1, max_k)
  p1_exps = p1_exps[1:]
  p2_exps = p2_exps[1:]
  
  plot_folder = "muzero_plots/goofspiel_" + str(cards) + "_" + points_order + "/"
  os.makedirs(plot_folder, exist_ok=True)  
  
  plt.plot(xs, p1_exps, label="P1 exploitability")
  plt.plot(xs, p2_exps, label="P2 exploitability")
  plt.xlabel("K")
  plt.ylabel("Exploitability")
  plt.legend()
  plt.savefig(plot_folder + "changed_spiel_kmeans_exploitability.png")
  
  
  
  
if __name__ == "__main__":
  main()