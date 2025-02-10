from jax_leduc import JaxLeduc
import jax
import jax.numpy as jnp
import numpy as np

from pyinstrument import Profiler


def test_gameplay(key, game:JaxLeduc):
  game_state, key, legals = game.initialize_structures(key)
  all_actions = jnp.arange(legals.shape[1])
  print("Dealt cards: ", game_state.private_cards)
  opp_action = 0
  acting_player = 0
  turn = 0
  terminal = False
  while not terminal:
    state_tensor, p1_iset_tensor, p2_iset_tensor, public_state_tensor = game.get_info(game_state)
    #print("State: ", state_tensor)
    #print("P1 iset: ", p1_iset_tensor)
    #print("P2_iset: ", p2_iset_tensor)
    #print("Public state: ", public_state_tensor)
    acting_legals = np.asarray(legals[acting_player], dtype="float64")
    pl_action = np.random.choice(all_actions, p=acting_legals / np.sum(acting_legals))
    actions = jnp.stack([pl_action, opp_action], axis=0) if acting_player == 0 else jnp.stack([opp_action, pl_action], axis=0)
    print("Actions: ", actions)
    print("Legal actions: ")
    print(legals)
    prev_public_card = game_state.public_card
    game_state, key, terminal, rewards, new_legals= game.apply_action(game_state, key, turn, actions)
    
    print("Public card: ", game_state.public_card)
    print("Current chips: ", game_state.current_chips)
    print("Rewards: ", rewards)
    legals = new_legals
    acting_player = 0 if prev_public_card == 0 and game_state.public_card > 0 else 1 - acting_player
    turn += 1

def choose_action(key, all_actions, acting_legals, acting_player):
  pl_action = jax.random.choice(key, all_actions, p=acting_legals / jnp.sum(acting_legals))
  actions = jnp.stack([pl_action, 0], axis=0) if acting_player == 0 else jnp.stack([0, pl_action], axis=0)
  key = jax.random.split(key, 1)[0]
  return key, actions


def test_batch(key, game:JaxLeduc, batch_size, experiment_repeats=20):
  vectorized_apply_action = jax.vmap(game.apply_action, in_axes=(0, 0, None, 0), out_axes=(0, 0, 0, 0, 0))
  vectorized_choice = jax.vmap(choose_action, in_axes=(0, 0, 0, None), out_axes=(0, 0))
  vectorized_init = jax.vmap(game.initialize_structures, in_axes=(0), out_axes=(0, 0, 0))
  profiler = Profiler()
  turn =  0
  acting_player = 0
  profiler.start()
  for _ in range(experiment_repeats):
    keys = jax.random.split(key, batch_size)
    game_state, keys, legals = vectorized_init(keys)
    all_actions = jnp.tile(jnp.arange(legals.shape[2]), (batch_size, 1))
    for _ in range(8):
      acting_legals = np.asarray(legals[:, acting_player, :], dtype="float64")
      keys, actions = vectorized_choice(keys, all_actions, acting_legals, acting_player)
      game_state, keys, terminal, rewards, new_legals = vectorized_apply_action(game_state, keys, turn, actions)
      legals = new_legals
      acting_player = 1- acting_player
      turn += 1
  profiler.stop()
  print(profiler.output_text(unicode=True, color=True))
    
   

def main():
  key = jax.random.key(42)
  game = JaxLeduc()
  test_gameplay(key, game)
  #test_batch(key, game, batch_size=10, experiment_repeats=20)
  



if __name__ == "__main__":
  main()

