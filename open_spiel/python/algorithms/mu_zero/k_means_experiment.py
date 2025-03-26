import scipy
import os
import numpy as np
from scipy.cluster.vq import kmeans, vq, whiten
import scipy.stats as st
import matplotlib.pyplot as plt
import jax
import pyspiel
import matplotlib.pyplot as plt
 
from open_spiel.python.algorithms.mu_zero.jax_games.jax_game import JaxGame, JaxPolicy
from open_spiel.python.algorithms.mu_zero.jax_games.jax_goofspiel import JaxGoofspiel, JaxModifiedGoofspiel
from open_spiel.python.algorithms.mu_zero.jax_games.jax_game_algorithms import nash_equilibrium_jax_game, exploitability_jax_game, prepare_cfr_from_game, extract_policy_from_cfr


from open_spiel.python.algorithms.mu_zero.mu_zero_cfr import MuZeroCFR, MuZeroCFRConstants
from open_spiel.python.algorithms.mu_zero.mu_zero_gameplay import convert_depth_to_jax, convert_player_depth_to_jax

from open_spiel.python.algorithms.mu_zero.experiments.utils import stringify, destringify


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
  # if random_seed is not None:
  #   np.random.seed(random_seed)

  # Make a copy of the data to avoid modifying the original
  data_copy = np.array(data, dtype=np.float64)

  # Normalize/whiten the data if requested
  # if normalize:
  #   data_copy = whiten(data_copy)

  # Perform K-means clustering
  centroids, distortion = kmeans(data_copy, k, seed=random_seed)

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


def get_all_public_states_with_isets_and_similarites(game: JaxGame, sim_type: str, policy_dict: dict) -> tuple[dict, dict]:


  assert sim_type == "legal" or sim_type == "policy", "Invalid similarity type"

 
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
    
    ps = np.array(ps)
    p1_iset_tensor = np.array(p1_iset_tensor)
    p2_iset_tensor = np.array(p2_iset_tensor)
    ps = stringify(ps)
    p1_iset = stringify(p1_iset_tensor)
    p2_iset = stringify(p2_iset_tensor)
    
    if ps not in state_infoset_map:
      state_infoset_map[ps] = [[], []]
      state_similarity_map[ps] = [[], []]
      
    if p1_iset not in state_infoset_map[ps][0]:
      state_infoset_map[ps][0].append(p1_iset)
      if sim_type == "policy":
        state_similarity_map[ps][0].append(policy_dict[p1_iset])
      elif sim_type == "legal":
        state_similarity_map[ps][0].append(np.array(legal_actions[0]))
      elif sim_type =="iset":
        state_similarity_map[ps][0].append(p1_iset_tensor)
      
    if p2_iset not in state_infoset_map[ps][1]:
      state_infoset_map[ps][1].append(p2_iset)
      
      if sim_type == "policy":
        state_similarity_map[ps][1].append(policy_dict[p2_iset])
      elif sim_type == "legal":
        state_similarity_map[ps][1].append(np.array(legal_actions[1]))
      elif sim_type =="iset":
        state_similarity_map[ps][1].append(p2_iset_tensor)
       
    
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

def state_string(ps, iset1, iset2):
  return stringify(ps) + "|" + stringify(iset1) + "|" + stringify(iset2)

# This uses old MuZero that cannot handle different legal actions in the same infoset
def prepare_cfr_from_clusters(game, cluster_map):
  
  possible_dynamics = {} # (iset1, iset2) -> (action1, action2) -> list(next_iset1, next_iset2, reward, terminal)
  dynamics = {} # (iset1, iset2) -> (action1, action2) -> (next_iset1, next_iset2, reward, terminal)
  # print(model.get_both_similarities_and_probs(init_ps, init_p1_isets, init_p2_iset)[2:4])
  def _traverse_tree(game_state, legal_actions, key, depth=0):
    state, p1_iset_tensor, p2_iset_tensor, ps = game.get_info(game_state) 
    ps = np.array(ps)
    p1_iset_tensor = np.array(p1_iset_tensor)
    p2_iset_tensor = np.array(p2_iset_tensor)
    
    p1_iset_str = stringify(p1_iset_tensor)
    p2_iset_str = stringify(p2_iset_tensor)
    
    abstracted_p1_iset = cluster_map[p1_iset_str]
    abstracted_p2_iset = cluster_map[p2_iset_str]
    
    state_str = state_string(ps, abstracted_p1_iset, abstracted_p2_iset)
     
    if state_str not in dynamics:
      dynamics[state_str] = {}
      
    
    for a1i, a1 in enumerate(legal_actions[0]):
      if a1 < 0.5:
        continue
      for a2i, a2 in enumerate(legal_actions[1]):
        if a2 < 0.5:
          continue
        
        next_key, action_key = jax.random.split(key)
        new_game_state, new_terminal, new_rewards, new_legals = game.apply_action(
            game_state, action_key, depth, np.array([a1i, a2i]))
        new_terminal = np.array([new_terminal])[0]
        new_rewards = np.array([new_rewards])[0]
        new_state, new_p1_iset_tensor, new_p2_iset_tensor, new_ps = game.get_info(new_game_state) 
        
        new_ps = np.array(new_ps)
        new_p1_iset_tensor = np.array(new_p1_iset_tensor)
        new_p2_iset_tensor = np.array(new_p2_iset_tensor)
        
        new_p1_iset_str = stringify(new_p1_iset_tensor)
        new_p2_iset_str = stringify(new_p2_iset_tensor)
    
    
        if new_terminal:
          new_ps = np.zeros(new_ps.shape[0])
          abstracted_new_p1_iset = np.zeros(abstracted_p1_iset.shape[0])
          abstracted_new_p2_iset = np.zeros(abstracted_p2_iset.shape[0])
        else:
          abstracted_new_p1_iset = cluster_map[new_p1_iset_str]
          abstracted_new_p2_iset = cluster_map[new_p2_iset_str]
    
        # new_state_str = new_p1_iset_str + new_p2_iset_str
        
        action = (a1i, a2i)
        if action not in dynamics[state_str]:
          dynamics[state_str][action] = (new_ps, abstracted_new_p1_iset, abstracted_new_p2_iset, new_rewards, new_terminal, 1)
          
        else:
          ps_dynamics, p1_dynamics, p2_dynamics, reward_dynamics, terminal_dynamics, visited_dynamics = dynamics[state_str][action]
           
          
          assert terminal_dynamics == new_terminal, "Should lead to terminal. Only true for goofspiel."
          assert np.all(ps_dynamics == new_ps)
          if  not np.all(p1_dynamics == abstracted_new_p1_iset):
            if state_str not in possible_dynamics:
              possible_dynamics[state_str] = {}
            if action not in possible_dynamics[state_str]:
              possible_dynamics[state_str][action] = []
            possible_dynamics[state_str][action].append((new_ps, abstracted_new_p1_iset, abstracted_new_p2_iset, new_rewards, new_terminal))
          if  not np.all(p2_dynamics == abstracted_new_p2_iset):
            if state_str not in possible_dynamics:
              possible_dynamics[state_str] = {}
            if action not in possible_dynamics[state_str]:
              possible_dynamics[state_str][action] = []
            possible_dynamics[state_str][action].append((new_ps, abstracted_new_p1_iset, abstracted_new_p2_iset, new_rewards, new_terminal))
          # assert np.all(p1_dynamics == abstracted_new_p1_iset)
          # assert np.all(p2_dynamics == abstracted_new_p2_iset)
          new_visited_dynamics = visited_dynamics + 1
          new_dynamics_reward = reward_dynamics + (1/new_visited_dynamics) * (new_rewards - reward_dynamics)
          dynamics[state_str][action] = (ps_dynamics, p1_dynamics, p2_dynamics, new_dynamics_reward, terminal_dynamics, new_visited_dynamics)
        
        if new_terminal:
          continue 
        
        _traverse_tree(new_game_state, new_legals, next_key, depth + 1)

  key = jax.random.key(0)
  state_key, init_key = jax.random.split(key)
  game_state, legals = game.initialize_structures(init_key)
  
  init_state, init_p1_iset, init_p2_iset, init_ps = game.get_info(game_state)
  
  init_ps = np.array(init_ps)
  init_p1_iset = cluster_map[stringify(np.array(init_p1_iset))]
  init_p2_iset = cluster_map[stringify(np.array(init_p2_iset))]
  
  _traverse_tree(game_state, legals, state_key)
  
  depth_iset_map  = [] # ID -> Public State concatenate with similarity
  depth_iset_legal  = []

  depth_history_action_utility  = [] # Float[D, H(D), A1, A2]
  depth_history_iset  = [] # Int[D, Pl, H(D)]
  depth_history_actions  = [] # Int[D, Pl, H(D), A] Just indices
  depth_history_legal = [] # Bool[D, Pl, H(D), A] or [D, H(D), A1, A2]

  depth_history_next_history = [] # Int[D, H(D), A1, A2]
  
  def _prepare_cfr_constants(ps, p1_iset, p2_iset, depth=0):
    actions = legals.shape[-1]
    # state = np.concatenate((ps, p1_iset, p2_iset))
    
    # mapped_p1_iset = np.concatenate((ps, p1_iset))
    # mapped_p2_iset = np.concatenate((ps, p2_iset))
    
    state_str = state_string(ps, p1_iset, p2_iset)
    
    if len(depth_iset_map) <= depth:
      depth_iset_map.append([[], []])
      depth_iset_legal.append([[], []])
      depth_history_action_utility.append([])
      depth_history_iset.append([[], []])
      depth_history_actions.append([[], []])
      depth_history_legal.append([])
      depth_history_next_history.append([])
      
    p1_iset_id = len(depth_iset_map[depth][0])
    p2_iset_id = len(depth_iset_map[depth][1])
    for p1_map_iset_id, p1_map_iset in enumerate(depth_iset_map[depth][0]):
      if np.all(p1_map_iset == p1_iset):
        p1_iset_id = p1_map_iset_id
        break
    for p2_map_iset_id, p2_map_iset in enumerate(depth_iset_map[depth][1]):
      if np.all(p2_map_iset == p2_iset):
        p2_iset_id = p2_map_iset_id
        break
    
    legal_actions = np.zeros((actions, actions))
    for (a1, a2) in dynamics[state_str].keys():
      legal_actions[a1, a2] = 1
      
    if p1_iset_id == len(depth_iset_map[depth][0]):
      depth_iset_map[depth][0].append(p1_iset)
      depth_iset_legal[depth][0].append(np.sum(legal_actions, 1) > 0)
    if p2_iset_id == len(depth_iset_map[depth][1]):
      depth_iset_map[depth][1].append(p2_iset)
      depth_iset_legal[depth][1].append(np.sum(legal_actions, 0) > 0)
    
    
    depth_history_iset[depth][0].append(p1_iset_id)
    depth_history_iset[depth][1].append(p2_iset_id)
    depth_history_actions[depth][0].append(np.arange(actions) + p1_iset_id * actions)
    depth_history_actions[depth][1].append(np.arange(actions) + p2_iset_id * actions)
    
    depth_history_legal[depth].append(legal_actions)
    depth_history_action_utility[depth].append(np.zeros((actions, actions)))
    depth_history_next_history[depth].append(np.full((actions, actions), -1))
    
    assert state_str in dynamics
    for (a1, a2), (next_ps, next_iset1, next_iset2, next_reward, next_terminal, _) in dynamics[state_str].items():
      depth_history_action_utility[depth][-1][a1, a2] = next_reward
      
      if next_terminal:
        continue
      
      next_history_id = 0
      if len(depth_history_iset) > depth + 1:
        next_history_id = len(depth_history_iset[depth+1][0])
      depth_history_next_history[depth][-1][a1, a2] = next_history_id 
      _prepare_cfr_constants(next_ps, next_iset1, next_iset2, depth + 1)
      pass
  
  
  _prepare_cfr_constants(init_ps, init_p1_iset, init_p2_iset)
  
  
  constants = MuZeroCFRConstants(
    resolving_player = 0,
    init_reaches = np.ones((2, 1)),
    depth_actions = [a[0][0].shape[0] for a in depth_history_actions],
    depth_iset_legal = convert_player_depth_to_jax(depth_iset_legal),
    
    depth_history_action_utility = convert_depth_to_jax(depth_history_action_utility),
    depth_history_iset = convert_depth_to_jax(depth_history_iset),
    depth_history_actions = convert_depth_to_jax(depth_history_actions),
    depth_history_legal = convert_depth_to_jax(depth_history_legal),
    
    depth_history_next_history = convert_depth_to_jax(depth_history_next_history),

  )
  
  cfr = MuZeroCFR(constants, depth_iset_map)
  return cfr
  
 
def compute_nash_kmeans_original_tree(game, cluster_map):
  muzero = prepare_cfr_from_game(game, cluster_map) 
  muzero.multiple_steps(2000) 
  policy_dict = extract_policy_from_cfr(game, muzero, cluster_map) 
  return policy_dict
  
def compute_nash_kmeans_modified_tree(game, cluster_map):
  muzero = prepare_cfr_from_clusters(game, cluster_map)
  muzero.multiple_steps(2000)
  policy_dict = extract_policy_from_cfr(game, muzero, cluster_map)
  return policy_dict

def print_policy(game:JaxGame, policy: JaxPolicy):
  

  def _traverse_tree(game_state, legal_actions, key, history = "", depth=0):
    
    state, p1_iset, p2_iset, ps = game.get_info(game_state)
    # ps_str = stringify(ps)
    p1_iset = np.array(p1_iset)
    p2_iset = np.array(p2_iset)
    p1_iset_str = stringify(p1_iset)
    p2_iset_str = stringify(p2_iset)
    
    print(history)
    print(policy[p1_iset_str])
    print(policy[p2_iset_str]) 
    
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
        # compu
        new_history = history + str(a1i) + " " + str(a2i) + " | "
        _traverse_tree(new_game_state, new_legals, next_key, new_history, depth + 1)
  
  start_key = jax.random.key(0)
  start_key, action_key = jax.random.split(start_key)
  # game_state, legals = game.initialize_structures(start_key)
  _traverse_tree(*game.initialize_structures(start_key), action_key)
  
def plot_all_cluster():
  orig_goof = True
  turns = 3
  first_card = 6
  orig_goof_experiments = [(3, 4), (4, 9), (5, 31)]
  # orig_goof_experiments = [(3, 4), (4, 9)]
  amount_seeds = 5
  for cards, max_k in orig_goof_experiments:
    
    exploitability_original_tree = np.zeros((2, max_k))
    exploitability_modified_tree = np.zeros((2, max_k))
    # exploitability_original_tree_confidence = np.zeros(2, max_k, 2)
    # exploitability_modified_tree_confidence = np.zeros(2, max_k, 2)
    
    
    if orig_goof:
      points_order = "descending"
      game = JaxGoofspiel(cards, points_order)
    else:
      points_order = "last_max"
      game = JaxModifiedGoofspiel(cards, turns, first_card) 
    _, dict_nash, nash_value = nash_equilibrium_jax_game(game)
    print(f"Nash of Goofspiel with {cards} cards and point card {points_order}: {nash_value[0]}")
    
    for sim_type in ["legal", "policy"]:
      state_iset_map, state_sim_map = get_all_public_states_with_isets_and_similarites(game, sim_type, dict_nash) 
      for k in range(1, max_k):
        print(k, flush=True)
        seed_data = np.zeros((4, amount_seeds))
        for seed in range(amount_seeds):
          state_cluster_map = {} 
          
          for state, sims in state_sim_map.items():
            ps = destringify(state) 
            for pl in range(2):
              pl_sims = np.array(sims[pl]) 
              cluster_amount = min(k, pl_sims.shape[0])
              center, labels, distance = perform_kmeans(pl_sims, cluster_amount, normalize=False, random_seed=seed)
              for iset_id, iset in enumerate(state_iset_map[state][pl]):
                
                state_cluster_map[iset] = np.concatenate((ps, center[labels[iset_id]])) 
          original_tree_policy = compute_nash_kmeans_original_tree(game, state_cluster_map)
          modified_tree_policy = compute_nash_kmeans_modified_tree(game, state_cluster_map)
          
          _, _, original_p1_exp, original_p2_exp = exploitability_jax_game(game, original_tree_policy) 
          _, _, modified_p1_exp, modified_p2_exp = exploitability_jax_game(game, modified_tree_policy) 
          seed_data[0, seed] = original_p1_exp
          seed_data[1, seed] = original_p2_exp
          seed_data[2, seed] = modified_p1_exp
          seed_data[3, seed] = modified_p2_exp
          
        
        exploitability_original_tree[0, k] = np.mean(seed_data[0])
        exploitability_original_tree[1, k] = np.mean(seed_data[1])
        exploitability_modified_tree[0, k] = np.mean(seed_data[2])
        exploitability_modified_tree[1, k] = np.mean(seed_data[3])
       
      xs = np.arange(1, max_k)
      plot_folder = "muzero_plots/goofspiel_" + str(cards) + "_" + points_order + "/"
      os.makedirs(plot_folder, exist_ok=True)  
      
      plt.plot(xs, exploitability_original_tree[0, 1:], label="P1 exploitability")
      plt.plot(xs, exploitability_original_tree[1, 1:], label="P2 exploitability")
      plt.xlabel("K")
      plt.ylabel("Exploitability")
      plt.legend()
      plt.savefig(plot_folder + sim_type +  "_original_tree_exploitability.png")
      plt.close()
      plt.cla()
      
      plt.plot(xs, exploitability_modified_tree[0, 1:], label="P1 exploitability")
      plt.plot(xs, exploitability_modified_tree[1, 1:], label="P2 exploitability")
      plt.xlabel("K")
      plt.ylabel("Exploitability")
      plt.legend()
      plt.savefig(plot_folder + sim_type +  "_modified_tree_exploitability.png")
      plt.close()
      plt.cla()
      
      
      
      
      
def main():
  import sys
  cards = 3
  turns = 3
  
  first_card = 6
  # for first_card in range(0, cards):
  points_order = "last_max"
  
  game = JaxModifiedGoofspiel(cards, turns, first_card) 
  
  points_order = "descending"
  game = JaxGoofspiel(cards, points_order)
  
  _, dict_nash, nash_value = nash_equilibrium_jax_game(game)
  print(nash_value)
  
  init_state, init_p1_iset, init_p2_iset, init_ps = game.get_info(game.initialize_structures(jax.random.key(0))[0])
  
  init_state = np.array(init_state)
  init_p1_iset = np.array(init_p1_iset)
  init_p2_iset = np.array(init_p2_iset)
  
  # print(dict_nash[stringify(init_p1_iset)])
  # print(nash_value)
  
  state_iset_map, state_sim_map = get_all_public_states_with_isets_and_similarites(game, "policy", dict_nash)
    
  max_k = 10
  p1_exps, p2_exps= np.zeros(max_k), np.zeros(max_k)
  
  for k in range(1, max_k):
    print(k, flush=True)
    state_cluster_map = {}
    # state_cluster_label_map = {}
    for state, sims in state_sim_map.items():
      ps = destringify(state)
      state_cluster_map[state] = [{}, {}]
      # state_cluster_label_map[state] = [{}, {}]
      for pl in range(2):
        pl_sims = np.array(sims[pl]) 
        cluster_amount = min(k, pl_sims.shape[0])
        center, labels, distance = perform_kmeans(pl_sims, cluster_amount, normalize=False, random_seed=7)
        for iset_id, iset in enumerate(state_iset_map[state][pl]):
          
          state_cluster_map[iset] = np.concatenate((ps, center[labels[iset_id]]))
          # state_cluster_map[iset] = center[labels[iset_id]]
          # state_cluster_map[state][pl][iset] = center[labels[iset_id]]
          # state_cluster_label_map[state][pl][iset] = labels[iset_id]  
    # policy_dict = compute_nash_kmeans_original_tree(game, state_cluster_map) 
    policy_dict = compute_nash_kmeans_modified_tree(game, state_cluster_map)
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
  plt.savefig(plot_folder + "_policy_kmean_symmetric_exploitability.png")
  
  
  
  
if __name__ == "__main__":
  plot_all_cluster()