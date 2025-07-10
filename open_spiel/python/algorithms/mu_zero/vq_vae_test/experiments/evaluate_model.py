
from argparse import ArgumentParser
import numpy as np
import jax
import os
import jax.numpy as jnp

from open_spiel.python.algorithms.mu_zero.vq_vae_test.train_utils import get_reference_policy
from open_spiel.python.algorithms.mu_zero.vq_vae_test.vq_vae_train import VQ_VAETrain, VQ_VAEConfig
from open_spiel.python.algorithms.mu_zero.vq_vae_test.point_card_matching import PointCardMatching

parser = ArgumentParser()
parser.add_argument("--model_path", type=str, default="trained_networks/point_card_matching3/seed99/network_seed42", help="Path to the trained model.")
parser.add_argument("--restore_step", type=int, default= -1, help="Which model step to restore. -1 if last saved step.")

parser.add_argument("--num_cards", type=int, default=3, help="Number of cards for the point card matching game. Make sure this matches the amount of cards of the stored model.")

def check_policies(model: VQ_VAETrain, game: PointCardMatching, eps=1e-3):
  """Traverse the real game tree and for each state first 
  represent it as afterstate and then check learned policy for that afterstate
  and compare it with reference"""
  dummy_key = jax.random.key(0)
  def _traverse_tree(state, legals, depth=0):
    #only interested in policy for player 1 here for reasons below
    state_reference_pols = np.asarray(get_reference_policy(state, legals)[0])
    state_tensor, _, _, _ = game.get_info(state)
    state_learned_pols = model.networks.get_policy_from_real(state_tensor)
    if np.max(np.abs(state_reference_pols - state_learned_pols)) >= eps:
      print(f"Policies differ by more than {eps} in state: {state}")
      print(f"Reference policy: {state_reference_pols}")
      print(f"Learned policy: {state_learned_pols}")
    #Only player 1 acts in PointCardMatching, the second player
    # has only one invalid action, so it can be viewed as a simultaneous move 
    # game for consistency with other JaxGames
    for ai, a in enumerate(state_reference_pols):
      if a < eps:
        continue
      next_state, next_legals, next_rewards, terminal = game.apply_action(state, dummy_key, depth, jnp.asarray([ai, 0]))
      if terminal:
        continue
      _traverse_tree(next_state, np.asarray(next_legals), depth + 1)
  init_state, init_legals = game.initialize_structures(dummy_key)
  _traverse_tree(init_state, np.asarray(init_legals))

def check_afterstate_tree(model: VQ_VAETrain, game:PointCardMatching, eps=1e-3):
  """Traverse the afterstate tree and check against reference policies
  obtained through traversing the original tree as well. Mainly intended to test the dynamics"""
  dummy_key = jax.random.key(0)
  actions = game.num_distinct_actions()
  def _traverse_tree(state, afterstate, legals, depth=0):
    afterstate_policy_logits, outcome = model.networks._jit_get_policy_outcome(afterstate)
    afterstate_policy = jax.nn.softmax(afterstate_policy_logits)
    afterstate_policy = np.asarray(afterstate_policy)
    state_reference_pols = np.asarray(get_reference_policy(state, legals)[0])
    if np.max(np.abs(state_reference_pols - afterstate_policy)) >= eps:
      print(f"Policies differ by more than {eps} in state: {state}")
      print(f"Reference policy: {state_reference_pols}")
      print(f"Learned policy: {afterstate_policy}")
    for ai, a in enumerate(state_reference_pols):
      if a < eps:
        continue
      next_state, next_legals, next_rewards, terminal = game.apply_action(state, dummy_key, depth, jnp.asarray([ai, 0]))
      outcome = jax.nn.one_hot(ai, actions)
      next_afterstate = model.networks._jit_get_next_afterstate(afterstate, outcome)
      if terminal:
        continue
      _traverse_tree(next_state, next_afterstate, np.asarray(next_legals), depth + 1)

  init_state, init_legals = game.initialize_structures(dummy_key)
  init_state_tensor, _, _, _ = game.get_info(init_state)
  init_afterstate = model.networks._jit_get_representation(init_state_tensor)
  _traverse_tree(init_state, init_afterstate, np.asarray(init_legals))

def afterstate_walk_test(model:VQ_VAETrain, game:PointCardMatching):
  """Does not check with reference directly, just perform a 
   test of getting afterstate outcomes and print out policies and given outcomes."""
  dummy_key = jax.random.key(0)
  state, legals = game.initialize_structures(dummy_key)
  init_state_tensor, _, _, _ = game.get_info(state)
  afterstate = model.networks._jit_get_representation(init_state_tensor)
  decoded_afterstate = model.networks._jit_decoder(afterstate)
  print(f"Init decoded afterstate: {decoded_afterstate}")
  print(f"Init state tensor: {init_state_tensor}")
  for i in range(model.config.trajectory_max):
    policy, outcome = model.networks._jit_get_policy_outcome(afterstate)
    print(f"Policy: {jax.nn.softmax(policy)}")
    print(f"Argmax outcome {outcome}")
    afterstate = model.networks._jit_get_next_afterstate(afterstate, outcome)
    decoded_afterstate = model.networks._jit_decoder(afterstate)
    ai = np.argmax(np.cumsum(outcome))
    state, legals, reward, terminal = game.apply_action(state, dummy_key, i, jnp.asarray([ai, 0]))
    #The model is not trained on terminal states
    if terminal:
      break
    state_tensor, _, _, _ = game.get_info(state)
    decoded_afterstate = model.networks._jit_decoder(afterstate)
    print(f"Next decoded afterstate: {decoded_afterstate}")
    print(f"Next state tensor {state_tensor}")




def main():
  args = parser.parse_args()
  #create a dummy config just for init before restoring the 
  #one from the checkpoint
  dummy_config = VQ_VAEConfig()
  model_path = args.model_path
  #Make sure that the given path is an absolute path
  if not model_path.startswith("/"):
    model_path = os.getcwd() + "/" + model_path
  model = VQ_VAETrain(dummy_config, args.num_cards, model_save_dir=model_path)
  model.restore_latest_checkpoint(args.restore_step)
  assert isinstance(model.game, PointCardMatching), "This test assumes that the model is trained on the PointCardMatching game, which it is not!"
  print("Real tree mapping test: ")
  check_policies(model, model.game)
  print("Dynamics test: ")
  check_afterstate_tree(model, model.game)
  print("Afterstate tree walk test: ")
  afterstate_walk_test(model, model.game)

if __name__ == "__main__":
  main()