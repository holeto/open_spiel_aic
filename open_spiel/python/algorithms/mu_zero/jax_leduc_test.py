from jax_leduc import JaxOriginalLeduc
import jax
import jax.numpy as jnp
import numpy as np


def test_gameplay(key, game:JaxOriginalLeduc):
  action_history, public_card,  private_cards, current_chips, key, round, turns_this_round, init_reaches, legals = game.initialize_structures(key)
  info = (action_history, public_card, private_cards)
  action_info = (*info, current_chips, key, round, turns_this_round)
  all_actions = jnp.arange(legals.shape[1])
  print("Dealt cards: ", private_cards)
  opp_action = 0
  acting_player = 0
  turn = 0
  terminal = False
  round = 0
  turns_this_round = 0
  while not terminal:
    state_tensor, p1_iset_tensor, p2_iset_tensor, public_state_tensor = game.get_info(*info)
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
    action_history, public_card, private_cards, current_chips, key, round, turns_this_round, terminal, rewards, new_legals= game.apply_action(*action_info, turn, actions)
    info = (action_history, public_card, private_cards)
    action_info = (*info, current_chips, key,  round, turns_this_round)
    
    print("Public card: ", public_card)
    print("Current chips: ", current_chips)
    print("Rewards: ", rewards)
    legals = new_legals
    acting_player = 1- acting_player
    turn += 1

def test_batch(key, game:JaxOriginalLeduc, batch_size):
  action_history, public_card,  private_cards, current_chips, key, round, turns_this_round, init_reaches, legals = game.initialize_structures(key, batch=batch_size)
  print(action_history.shape)
  print(public_card.shape)  
  print(private_cards.shape)
  print(current_chips.shape)
  print(legals.shape)

def main():
  key = jax.random.key(42)
  game = JaxOriginalLeduc()
  test_gameplay(key, game)
  #test_batch(key, game, batch_size=20)
  



if __name__ == "__main__":
  main()

