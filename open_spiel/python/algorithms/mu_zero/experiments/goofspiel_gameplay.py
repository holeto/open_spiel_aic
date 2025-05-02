import argparse
import jax.numpy as jnp
import numpy as np
import time
import jax
#from joblib import Parallel, delayed
from copy import copy
from pyinstrument import Profiler

from open_spiel.python.algorithms.mu_zero.jax_games.jax_goofspiel import JaxGoofspiel
from open_spiel.python.algorithms.mu_zero.experiments.utils import load_model
from open_spiel.python.algorithms.mu_zero.mu_zero_train import MuZeroTrain
from open_spiel.python.algorithms.mu_zero.mu_zero_gameplay import MuZeroGameplay, MuZeroGameplayConfig
#from goofspiel_exploitability import stringify

parser = argparse.ArgumentParser()

# Training setting
parser.add_argument("--model_path", type=str, default="muzero_networks/goofspiel_13_descending/seed_111133/muzero_2.pkl", help="Model path") 
parser.add_argument("--opp_model_path", type=str, default="muzero_networks/goofspiel_13_descending/seed_111133/muzero_2.pkl", help="Opponent model path")
parser.add_argument("--depth_limit", type=int, default=1, help="Depth limit for exploitability calculation in each infoset")
parser.add_argument("--resolve_iterations", type=int, default=1000, help="Number of CFR iterations")

#Gameplay setting
parser.add_argument("--player", type=int, default=0, help="Resolving player")
parser.add_argument("--rounds", type=int, default=10, help="Number of rounds to play until the end.") 
parser.add_argument("--verbose", type=bool, default=True, help="Print played actions.")
parser.add_argument("--seed", type=int, default=43, help="Seed for the random number generator")


def get_rnad_policy(opp_iset, opp_legals, model:MuZeroTrain):
  pi = model._jit_get_policy(model.network_parameters.rnad_params_target, opp_iset, opp_legals)
  pi = np.asarray(pi, dtype="float64")
  pi /= np.sum(pi)
  return pi

def get_opponent_action(opponent:str, opp_iset, opp_legals, model:MuZeroTrain):
  #TODO: Add different opponents here
  pi = None
  if opponent == "rnad":
    pi = model._jit_get_policy(model.network_parameters.rnad_params_target, opp_iset, opp_legals)
    pi = np.asarray(pi, dtype="float64")
    pi /= np.sum(pi)
    #pi = opp_policy[stringify(opp_iset)]
  #random opponent
  else:
    pi = opp_legals
    pi = np.asarray(pi, dtype="float64")
    pi /= np.sum(pi)
  return np.random.choice(np.arange(model.game.cards), p=pi)

def play_single_round(args, model:MuZeroTrain, gameplay: MuZeroGameplay,parallel:bool = True):
  key = jax.random.key(0)
  key, init_key = jax.random.split(key)
  game_state, legals = model.game.initialize_structures(init_key)
  opp = 1 - args.player
  opp_legals = legals[opp]
  
  _, p1_iset, p2_iset, ps = model.game.get_info(game_state)
  turn = 0
  cumulative_reward = 0
  all_actions = np.arange(model.game.cards)
  temp_gameplay = copy(gameplay) if parallel else gameplay
  #temp_gameplay = copy(gameplay)
  
  #play until terminal
  for _ in range(model.game.cards - 1):
    #profiler.start()
    #start_time = time.time()
    pl_iset = p1_iset if args.player == 0 else p2_iset
    opp_iset = p2_iset if args.player == 0 else p1_iset
    pl_action = temp_gameplay.get_action(ps, pl_iset)
    opp_action = get_opponent_action(args.opponent, opp_iset, opp_legals, all_actions, model)
    actions = [[],[]]
    actions[args.player] = pl_action
    actions[opp] = opp_action
    if args.verbose:
      print("Applying action: ", actions)
    actions = jnp.stack(actions, axis=0)
    key, action_key = jax.random.split(key)
    game_state, terminal, rewards, legals = model.game.apply_action(game_state, action_key, turn, actions)
    cumulative_reward += rewards
    opp_legals = legals[opp]
    turn += 1
    _, p1_iset, p2_iset, ps = model.game.get_info(game_state)
    #print("Action choosing time: ", time.time() - start_time)
    #profiler.stop()
    #print(profiler.output_text(color=True, unicode=True))
  temp_gameplay.reset()
  if args.verbose:
    print("P1 reward: ", cumulative_reward)
    print("P2 reward: ", -cumulative_reward)
  return cumulative_reward


def sequential_experiment(args, model, muzero_gameplay):
  start_time = time.time()
  mean_reward = 0
  for _ in range(args.rounds):
    mean_reward += play_single_round(args, model, muzero_gameplay, parallel=False)
  mean_reward /= args.rounds
  print("Execution time: ", time.time() - start_time)
  print("P1 mean reward: ", mean_reward)
  print("P2 mean reward: ", -mean_reward)

def parallel_experiment(args, model, muzero_gameplay, njobs=8):
  assert False, "Do not call this function yet. It does not work!"
  if args.verbose:
    print("Warning! Parallel experiment was called with verbose prints.These are not thread safe.")
  start_time = time.time()
  cumulative_rewards = Parallel(n_jobs=njobs)(delayed(play_single_round)(args, model, muzero_gameplay) for _ in range(args.rounds))
  cumulative_rewards = np.asarray(cumulative_rewards)
  mean_reward = np.sum(cumulative_rewards) / cumulative_rewards.shape[0]
  print("Execution time: ", time.time() - start_time)
  print("P1 mean reward: ", mean_reward)
  print("P2 mean reward: ", -mean_reward)

def run_heads_against_rnad(args):
  model = load_model(args.model_path)
  opp_model = load_model(args.opp_model_path)
  assert isinstance(model.game, JaxGoofspiel)
  assert isinstance(opp_model.game, JaxGoofspiel)
  assert model.game.points_order == "descending"
  assert opp_model.game.points_order == "descending"
  assert model.game.cards == opp_model.game.cards
  assert model.game.points_order == opp_model.game.points_order
  assert model.game.points_order == opp_model.game.points_order 
  
  gp_config = MuZeroGameplayConfig(player=args.player,
                                   resolve_iterations=args.resolve_iterations,
                                   depth_limit=args.depth_limit)
  muzero_gameplay = MuZeroGameplay(model, gp_config)  
    
  
  rewards = []
  
  jax_key = jax.random.key(args.seed)
  np_key = np.random.RandomState(args.seed)
  
  player = args.player
  opponent = 1 - player
  for _ in range(args.rounds): 
    muzero_gameplay.reset()
    game_state, legals = model.game.initialize_structures(jax_key)
    terminal = False
    turn = 0
    reward = 0
    while not terminal:
      _, p1_iset, p2_iset, ps = model.game.get_info(game_state)
      
      if player == 0: 
        p1_policy = muzero_gameplay.get_policy(ps, p1_iset)
        p2_policy = get_rnad_policy(p2_iset, legals[1], model)
      else:
        p2_policy = muzero_gameplay.get_policy(ps, p2_iset)
        p1_policy = get_rnad_policy(p1_iset, legals[0], model)
      
      p1_action = np_key.choice(np.arange(model.game.cards), p=p1_policy)
      p2_action = np_key.choice(np.arange(model.game.cards), p=p2_policy)
      
      
      jax_key, action_key = jax.random.split(jax_key)
      actions = jnp.stack([p1_action, p2_action], axis=0)
      game_state, terminal, game_rewards, legals = model.game.apply_action(game_state, action_key, turn, actions)
      turn += 1
      terminal = np.array(terminal) 
      reward += np.array(game_rewards)
      
    print(reward)
    rewards.append(reward)
    
    
  print("Player mean reward: ", np.mean(rewards))
  print("Opponent mean reward: ", -np.mean(rewards))
  
def main(): 
  args = parser.parse_args()
  run_heads_against_rnad(args)

if __name__ == "__main__":
  main()