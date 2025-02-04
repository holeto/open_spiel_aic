from jax_leduc import JaxOriginalLeduc
import jax
import jax.numpy as jnp
import numpy as np

def get_next_rng_key(key):
  key, new_key = jax.random.split(key)
  return new_key

def main():
  key = jax.random.key(99)
  game = JaxOriginalLeduc()
  action_history, public_card, current_chips, key, init_reaches, private_cards, legals = game.initialize_structures(key)
  info = (action_history, public_card, current_chips)
  all_actions = jnp.arange(legals.shape[1])
  #print("Dealt cards: ", private_cards)
  opp_action = 0
  acting_player = 0
  turn = 0
  terminal = False
  round = 0
  turns_this_round = 0
  while not terminal:
    acting_legals = np.asarray(legals[acting_player], dtype="float64")
    key = get_next_rng_key(key)
    pl_action = np.random.choice(all_actions, p=acting_legals / np.sum(acting_legals))
    actions = jnp.stack([pl_action, opp_action], axis=0) if acting_player == 0 else jnp.stack([opp_action, pl_action], axis=0)
    #print(actions)
    #print(legals)
    action_history, public_card, current_chips, terminal, rewards, round, turns_this_round, key, new_legals = game.apply_action(*info, key, actions, round, turns_this_round, turn)
    info = (action_history, public_card, current_chips)
    #print(info)
    #print(rewards)
    legals = new_legals
    acting_player = 1- acting_player
    turn += 1



if __name__ == "__main__":
  main()

