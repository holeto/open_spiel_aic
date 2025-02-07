import pyspiel
import argparse
import chex
import jax
import time
import jax.numpy as jnp

from jax_leduc import JaxLeduc, LeducGameState
from game_test_utils import extract_from_spiel ,histogram, compare_hists
from pyinstrument import Profiler
from collections import deque

parser = argparse.ArgumentParser()

parser.add_argument("--seed", type=int, default=42, help= "PRNG key seed")

@chex.dataclass(frozen=True)
class SampleTrajectoryCarry:
  game_state: LeducGameState
  terminal: chex.Array
  key: chex.Array
  legal_actions: chex.Array


  

#Extract from jax leduc how
#many states fall under each infoset
#and how many infosets of each player
#fall for each public state
def extract_from_jax_leduc(seed):
  game = JaxLeduc()
  #state_tensors = []
  count_states_by_isets = [{} for _ in range(2)]
  count_isets_by_public_state = [{} for _ in range(2)]
  all_actions = jnp.tile(jnp.arange(4), (2, 1))
  key = jax.random.key(seed)
  keys = jax.random.split(key, 1)
  #represent state as the carry information from it
  vectorized_init = jax.vmap(game.initialize_structures, in_axes=(0), out_axes=(0, 0, 0))
  game_state, key, legals = vectorized_init(keys)
  vectorized_get_info = jax.vmap(game.get_info, in_axes=(0), out_axes=(0, 0, 0, 0))
  vectorized_apply_action = jax.vmap(game.apply_action, in_axes=(0, 0, None, 0), out_axes=(0, 0, 0, 0, 0))
  def get_new_carries(carry : SampleTrajectoryCarry, joint_action, turn):
    carries =[]
    public_cards = []
    game_state, key, terminal, rewards, new_legals  = vectorized_apply_action(carry.game_state, carry.key, turn, joint_action)
    if terminal:
      return []
    #simulate second chance node
    if carry.game_state.public_card[0] == 0 and game_state.public_card[0] > 0:
      for c in range(6):
        if c in game_state.private_cards[0]:
          continue
        public_cards.append(jnp.array(c + 1, dtype=int)[None, ...])
    else:
      public_cards.append(game_state.public_card)
    for pc in public_cards:
      new_game_state = LeducGameState( 
        action_history = game_state.action_history,
        public_card = pc,
        private_cards = game_state.private_cards,
        current_chips = game_state.current_chips,
        turns_this_round = game_state.turns_this_round
      )
      new_carry = SampleTrajectoryCarry(
        game_state = new_game_state,
        terminal= terminal,
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
      new_game_state = LeducGameState(
                        action_history = game_state.action_history,
                        public_card = game_state.public_card,
                        private_cards = private_cards,
                        current_chips = game_state.current_chips,
                        turns_this_round = game_state.turns_this_round
      )
      init_carry = SampleTrajectoryCarry(
                        game_state = new_game_state,
                        terminal= jnp.zeros([1,1], dtype=bool),
                        key = key,
                        legal_actions = legals)
      q.append((init_carry, 0))
  while len(q) > 0:
    carry, turn = q.popleft()
    if(carry.terminal):
      continue
    #print(carry.game_state.current_chips)
    state, p1_iset, p2_iset, public_state = vectorized_get_info(carry.game_state)
    #squeeze out the 1-element batch
    infosets_str = [jnp.array_str(p1_iset[0]), jnp.array_str(p2_iset[0])]
    public_state_str = jnp.array_str(public_state[0])
    for pl, iset_str in enumerate(infosets_str):
      if iset_str in count_states_by_isets[pl].keys():
        count_states_by_isets[pl][iset_str] += 1
      else:
        count_states_by_isets[pl][iset_str] = 1   
        if(public_state_str in count_isets_by_public_state[pl].keys()):
          count_isets_by_public_state[pl][public_state_str] += 1
        else:
          count_isets_by_public_state[pl][public_state_str] = 1
    if not str(state) in visited:
      legal_mask_p1 = carry.legal_actions[0][0].astype(bool)
      legal_mask_p2 = carry.legal_actions[0][1].astype(bool)
      p1_actions = all_actions[0][legal_mask_p1]
      p2_actions = all_actions[1][legal_mask_p2]
      visited.append(str(state))
      for a1 in p1_actions:
        for a2 in p2_actions:
          new_carries = get_new_carries(carry, jnp.asarray([a1, a2])[jnp.newaxis, ...], turn)
          for new_carry in new_carries:
            q.append((new_carry, turn + 1))
  return count_states_by_isets[0].values(), count_states_by_isets[1].values(), count_isets_by_public_state[0].values(), count_isets_by_public_state[1].values()



def main():
  args= parser.parse_args()
  profiler = Profiler()
  game = pyspiel.load_game("leduc_poker")
  #start_time = time.time()
  p1_iset_groups, p2_iset_groups, public_states_groups = extract_from_spiel(game)
  #print("Spiel extraction time: ", time.time() - start_time)
  #start_time = time.time()
  profiler.start()
  jax_p1_iset_groups, jax_p2_iset_groups, jax_p1_pub_state_isets, jax_p2_pub_state_isets = extract_from_jax_leduc(args.seed)
  profiler.stop()
  print(profiler.output_text(unicode=True, color=True))
  #print("Jax extraction time: ", time.time() - start_time)
  #print(len(p1_iset_groups))
  #print(len(jax_p1_iset_groups))
  p1_isets_hist = histogram(list(p1_iset_groups))
  jax_p1_isets_hist = histogram(list(jax_p1_iset_groups))
  num_states = 0
  for i, states in enumerate(p1_isets_hist):
    num_states += (i + 1) * states
  print("STATES NUMBER: ", num_states)
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
  jax_public_state_p1_hist = histogram(list(jax_p1_pub_state_isets))
  jax_public_state_p2_hist = histogram(list(jax_p2_pub_state_isets))
  print("PUBLIC STATES:")
  print("P1:")
  print(jax_public_state_p1_hist)
  print("P2:")
  print(jax_public_state_p2_hist)
  #compare_hists(public_state_hist, jax_public_state_hist)

if __name__ == "__main__":
  main()