import argparse
import jax.numpy as jnp
import numpy as np

from open_spiel.python.algorithms.mu_zero.jax_goofspiel import JaxOriginalGoofspiel
from open_spiel.python.algorithms.mu_zero.experiments.utils import load_model
from open_spiel.python.algorithms.mu_zero.mu_zero import MuZeroTrain
from open_spiel.python.algorithms.mu_zero.mu_zero_gameplay import MuZeroGameplay, MuZeroGameplayConfig
from goofspiel_exploitability import stringify

parser = argparse.ArgumentParser()

# Training setting
parser.add_argument("--model_path", type=str, default="muzero_networks/goofspiel_3_descending/seed_42/muzero_99.pkl", help="Model path") 
parser.add_argument("--depth_limit", type=int, default=1, help="Depth limit for exploitability calculation in each infoset")
parser.add_argument("--resolve_iterations", type=int, default=1000, help="Number of CFR iterations")

#Gameplay setting
parser.add_argument("--player", type=int, default=0, help="Resolving player")
parser.add_argument("--rounds", type=int, default=10, help="Number of rounds to play until the end.")
parser.add_argument("--opponent", type=str, default="random", help="Opponent strategy")
parser.add_argument("--verbose", type=bool, default=False, help="Print played actions.")


#Extracts full policy as dictionary for an RNaD opponent
# def extract_rnad_policy(model:MuZeroTrain, game:JaxOriginalGoofspiel, player):
#   policy = {}
#   visited_isets = {}
#   isets = []
#   iset_legals = []
#   def _extract_goofspiel_isets(states, depth: int = 0):
#     for state in states:
#       _, p1_iset, p2_iset, public_state = game.get_info(*state[:-1])
#       iset = p2_iset if player == 0 else p1_iset
#       iset_str = stringify(iset)
#       opp_legals = state[-1][1 - player]
#       if iset_str not in visited_isets:
#         iset_idx = len(isets)
#         visited_isets[iset_str] = iset_idx
#         isets.append(iset)
#         iset_legals.append(opp_legals)

#     if depth + 2 >= game.cards:
#       return
#     next_states = []
#     for state in states:
#       for a1i, a1 in enumerate(state[-1][0]):
#         if a1 < 0.5:
#           continue
#         for a2i, a2 in enumerate(state[-1][1]):
#           if a2 < 0.5:
#             continue
#           new_legals, new_rewards, new_point_cards, new_played_cards, new_p1_points = game.apply_action(*state[:-1], depth, np.array([a1i, a2i])) 
#           new_info = (new_point_cards, new_played_cards, new_p1_points, new_legals)
#           next_states.append(new_info)
#     _extract_goofspiel_isets(next_states, depth + 1)
#   init_info = game.initialize_structures()
#   _extract_goofspiel_isets([init_info])
#   isets = np.array(isets)
#   iset_legals = np.array(iset_legals)
#   pi = model._jit_get_policy(model.network_parameters.rnad_params_target, isets, iset_legals)
#   for iset, pols in zip(isets, pi):
#     normalized_pols = np.asarray(pols, dtype="float64")
#     normalized_pols /= np.sum(normalized_pols)
#     policy[stringify(iset)] = normalized_pols
#   return policy



def get_opponent_action(opponent:str, opp_iset, opp_legals, actions, model:MuZeroTrain):
  #TODO: Add different opponents here
  pi = None
  if opponent == "rnad":
    pi = model._jit_get_policy(model.network_parameters.rnad_params_target, opp_iset, opp_legals)
    pi = np.asarray(pi, dtype="float64")
    pi /= np.sum(pi)
    #pi = opp_policy[stringify(opp_iset)]
  #random opponent
  else:
    pi = np.ones_like(actions) * opp_legals
    pi /= np.sum(pi)
  return np.random.choice(actions, p=pi)

def play_single_round(args, muzero_gameplay: MuZeroGameplay, game: JaxOriginalGoofspiel, model:MuZeroTrain):
  init_info = game.initialize_structures()
  opp = 1 - args.player
  opp_legals = init_info[-1][opp]
  
  _, p1_iset, p2_iset, ps = game.get_info(*init_info[:-1])
  info = init_info[:-1]
  turn = 0
  cumulative_reward = 0
  all_actions = np.arange(game.cards)
  
  #play until terminal
  for _ in range(game.cards - 1):
    pl_iset = p1_iset if args.player == 0 else p2_iset
    opp_iset = p2_iset if args.player == 0 else p1_iset
    pl_action = muzero_gameplay.get_action(ps, pl_iset)
    opp_action = get_opponent_action(args.opponent, opp_iset, opp_legals, all_actions, model)
    actions = [[],[]]
    actions[args.player] = pl_action
    actions[opp] = opp_action
    if args.verbose:
      print("Applying action: ", actions)
    actions = jnp.stack(actions, axis=0)
    legals, rewards, point_cards, played_cards, p1_points = game.apply_action(*info, turn, actions)
    cumulative_reward += rewards
    info = (point_cards, played_cards, p1_points)
    opp_legals = legals[opp]
    turn += 1
    _, p1_iset, p2_iset, ps = game.get_info(*info)
  print("P1 reward: ", cumulative_reward)
  print("P2 reward: ", -cumulative_reward)


def main(): 
  args = parser.parse_args()
  model = load_model(args.model_path) 
  assert isinstance(model.game, JaxOriginalGoofspiel)
  assert model.game.points_order == "descending", "We cannot handle exploitability of different Goofspiels"
  gp_config =MuZeroGameplayConfig(player=args.player)
  muzero_gameplay = MuZeroGameplay(model, gp_config)
  #opp_policy = extract_rnad_policy(model, model.game, args.player) if args.opponent == "rnad" else None
  for _ in range(args.rounds):
    play_single_round(args, muzero_gameplay, model.game, model)
    muzero_gameplay.reset()
  


if __name__ == "__main__":
  main()