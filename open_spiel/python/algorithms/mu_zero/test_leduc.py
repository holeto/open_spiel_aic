import pyspiel
import argparse
import chex
import jax
import jax.numpy as jnp
from jax_leduc import JaxOriginalLeduc
from game_test_utils import extract_from_spiel ,histogram, compare_hists
from collections import deque

parser = argparse.ArgumentParser()

parser.add_argument("--seed", type=int, default=42, help= "PRNG key seed")

@chex.dataclass(frozen=True)
class SampleTrajectoryCarry:
  action_history: chex.Array
  public_card: chex.Array
  private_cards: chex.Array
  current_chips: chex.Array
  terminal: chex.Array
  round: chex.Array
  turns_this_round: chex.Array
  key: chex.Array
  legal_actions: chex.Array

  

#Extract from jax leduc how
#many states fall under each infoset/public state
def extract_from_jax_leduc(seed):
  game = JaxOriginalLeduc()
  #state_tensors = []
  count_states_by_isets = [{} for _ in range(2)]
  count_states_by_public_state = {}
  all_actions = jnp.tile(jnp.arange(4), (2, 1))
  key = jax.random.key(seed)
  #represent state as the carry information from it
  action_history, public_card,  chosen_private_cards, current_chips, key, round, turns_this_round, init_reaches, legals = game.initialize_structures(key,batch=1)
  vectorized_get_info = jax.vmap(game.get_info, in_axes=(0, 0, 0), out_axes=(0, 0, 0, 0))
  vectorized_apply_action = jax.vmap(game.apply_action, in_axes=(0, 0, 0, 0, 0, 0, 0, None, 0), out_axes=(0, 0, 0, 0, 0, 0, 0, 0, 0, 0))
  def get_new_carries(carry : SampleTrajectoryCarry, joint_action, turn):
    carries =[]
    public_cards = []
    carry_info = vectorized_apply_action(carry.action_history, carry.public_card, carry.private_cards, carry.current_chips, carry.key, carry.round, carry.turns_this_round, turn, joint_action)
    action_history, public_card, private_cards, current_chips, key, round, turns_this_round, terminal, rewards, new_legals = carry_info
    if terminal:
      return []
    #simulate second chance node
    if carry.public_card[0] == 0 and public_card[0] > 0:
      for c in range(6):
        if c in private_cards[0]:
          continue
        public_cards.append(jnp.array(c, dtype=int)[None, ...])
    else:
      public_cards.append(public_card)
    for pc in public_cards:
      new_carry = SampleTrajectoryCarry(
        action_history = action_history,
        public_card = pc,
        private_cards = private_cards,
        current_chips = current_chips,
        terminal= terminal,
        round = round,
        turns_this_round = turns_this_round,
        key = key,
        legal_actions = new_legals)
      carries.append(new_carry)
    return carries
  q = deque()
  visited = []
  #6 cards in total
  for c1 in range(6):
    for c2 in range(6):
      if c1 == c2:
        continue
      private_cards = jnp.array([c1, c2], dtype=int)[None, ...]
      init_carry = SampleTrajectoryCarry(action_history = action_history,
                        public_card = public_card,
                        private_cards = private_cards,
                        current_chips = current_chips,
                        terminal= jnp.zeros([1,1], dtype=bool),
                        round = round,
                        turns_this_round = turns_this_round,
                        key = key,
                        legal_actions = legals)
      q.append((init_carry, 0))
  while len(q) > 0:
    carry, turn = q.popleft()
    if(carry.terminal):
      continue
    #print(carry.current_chips[0])
    #print(carry.action_history[0])
    state, p1_iset, p2_iset, public_state = vectorized_get_info(carry.action_history, carry.public_card, carry.private_cards)
    #squeeze out the 1-element batch
    infosets_str = [jnp.array_str(p1_iset[0]), jnp.array_str(p2_iset[0])]
    public_state_str = jnp.array_str(public_state[0])
    if(public_state_str in count_states_by_public_state.keys()):
      count_states_by_public_state[public_state_str] += 1
    else:
      count_states_by_public_state[public_state_str] = 1
    for pl, iset_str in enumerate(infosets_str):
      if iset_str in count_states_by_isets[pl].keys():
        count_states_by_isets[pl][iset_str] += 1
      else:
        count_states_by_isets[pl][iset_str] = 1
    if not str(carry) in visited:
      legal_mask_p1 = carry.legal_actions[0][0].astype(bool)
      legal_mask_p2 = carry.legal_actions[0][1].astype(bool)
      #masked_actions = all_actions[legal_mask].reshape((2, -1))
      #print(carry.legal_actions[0])
      p1_actions = all_actions[0][legal_mask_p1]
      p2_actions = all_actions[1][legal_mask_p2]
      visited.append(str(carry))
      for a1 in p1_actions:
        for a2 in p2_actions:
          new_carries = get_new_carries(carry, jnp.asarray([a1, a2])[jnp.newaxis, ...], turn)
          for new_carry in new_carries:
            q.append((new_carry, turn + 1))
  return count_states_by_isets[0].values(), count_states_by_isets[1].values(), count_states_by_public_state.values()



def main():
  args= parser.parse_args()
  game = pyspiel.load_game("leduc_poker")
  p1_iset_groups, p2_iset_groups, public_states_groups = extract_from_spiel(game)
  jax_p1_iset_groups, jax_p2_iset_groups, jax_public_states_groups = extract_from_jax_leduc(args.seed)
  #print(len(p1_iset_groups))
  #print(len(jax_p1_iset_groups))
  p1_isets_hist = histogram(list(p1_iset_groups))
  jax_p1_isets_hist = histogram(list(jax_p1_iset_groups))
  print("P1 ISETS:")
  print("SPIEL:")
  print(p1_isets_hist)
  print("JAX")
  print(jax_p1_isets_hist)
  compare_hists(p1_isets_hist, jax_p1_isets_hist)

  p2_isets_hist = histogram(list(p2_iset_groups))
  jax_p2_isets_hist = histogram(list(jax_p2_iset_groups))
  print("P2 ISETS:")
  print("SPIEL:")
  print(p2_isets_hist)
  print("JAX")
  print(jax_p2_isets_hist)
  compare_hists(p2_isets_hist, jax_p2_isets_hist)
  #public_state_hist = histogram(list(public_states_groups))
  #jax_public_state_hist = histogram(list(jax_public_states_groups))
  #print("PUBLIC STATES:")
  #print("SPIEL:")
  #print(public_state_hist)
  #print("JAX")
  #print(jax_public_state_hist)
  #compare_hists(public_state_hist, jax_public_state_hist)

if __name__ == "__main__":
  main()