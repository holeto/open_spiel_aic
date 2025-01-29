import argparse
import jax.numpy as jnp
import numpy as np

from open_spiel.python.algorithms.mu_zero.jax_goofspiel import JaxOriginalGoofspiel
from open_spiel.python.algorithms.mu_zero.experiments.utils import load_model
from open_spiel.python.algorithms.mu_zero.mu_zero_gameplay import MuZeroGameplay, MuZeroGameplayConfig
from open_spiel.python.algorithms.mu_zero.mu_zero_cfr import MuZeroCFR
from open_spiel.python.algorithms.mu_zero.mu_zero_gameplay import prepare_cfr_structure, find_next_root

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


def get_opponent_action(opponent:str, opp_iset, opp_legals):
  opp_pols = np.ones_like(opp_legals, dtype="float64") * opp_legals
  #uniform strategy of random opponent
  if opponent == "random":
    opp_pols /= np.sum(opp_pols)
  #TODO: Add different opponents here
  return np.random.choice(opp_legals, p=opp_pols).astype(int)

def play_single_round(player: int, opponent:str, muzero_gameplay: MuZeroGameplay, game: JaxOriginalGoofspiel, verbose: bool):
  init_info = game.initialize_structures()
  opp = 1 - player
  opp_legals = init_info[-1][opp]
  
  _, p1_iset, p2_iset, ps = game.get_info(*init_info[:-1])
  info = init_info[:-1]
  turn = 0
  cumulative_reward = 0
  
  #play until terminal
  for _ in range(game.cards - 1):
    pl_iset = p1_iset if player == 0 else p2_iset
    opp_iset = p2_iset if player == 0 else p1_iset
    pl_action = muzero_gameplay.get_action(ps, pl_iset)
    #random opponent
    opp_action = get_opponent_action(opponent, opp_iset, opp_legals)
    actions = [[],[]]
    actions[player] = pl_action
    actions[opp] = opp_action
    if verbose:
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
  for _ in range(args.rounds):
    play_single_round(args.player, args.opponent, muzero_gameplay, model.game, args.verbose)
    muzero_gameplay.reset()
  


if __name__ == "__main__":
  main()