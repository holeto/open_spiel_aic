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
  key = jax.random.key(seed)
  #represent state as the carry information from it
  game_state, legals = game.initialize_structures(key)
  def get_new_carries(carry : SampleTrajectoryCarry, joint_action, turn):
    carries =[]
    public_cards = []
    game_state, terminal, reward, new_legals  = game.apply_action(carry.game_state, carry.key, turn, joint_action)
    if terminal:
      return []
    #simulate second chance node
    pc_chance_outcomes = game.generate_all_public_card_nodes(game_state)
    for outcome in pc_chance_outcomes:
      new_carry = SampleTrajectoryCarry(
        game_state = outcome,
        terminal= terminal,
        key = key,
        legal_actions = new_legals)
      carries.append(new_carry)
    return carries
  q = deque()
  visited = []
  #6 cards in total
  roots, legals = game.generate_all_private_card_nodes()
  for root_state in roots:
    init_carry = SampleTrajectoryCarry(
                      game_state = root_state,
                      terminal= jnp.zeros([1,1], dtype=bool),
                      key = key,
                      legal_actions = legals)
    q.append((init_carry, 0))
  while len(q) > 0:
    carry, turn = q.popleft()
    if(carry.terminal):
      continue
    #print(carry.game_state)
    state, p1_iset, p2_iset, public_state = game.get_info(carry.game_state)
    #squeeze out the 1-element batch
    infosets_str = [jnp.array_str(p1_iset), jnp.array_str(p2_iset)]
    public_state_str = jnp.array_str(public_state)
    for pl, iset_str in enumerate(infosets_str):
      if iset_str in count_states_by_isets[pl].keys():
        count_states_by_isets[pl][iset_str] += 1
      else:
        count_states_by_isets[pl][iset_str] = 1   
        if(public_state_str in count_isets_by_public_state[pl].keys()):
          count_isets_by_public_state[pl][public_state_str] += 1
        else:
          count_isets_by_public_state[pl][public_state_str] = 1
    if not jnp.array_str(state) in visited:
      legal_mask_p1 = carry.legal_actions[0].astype(bool)
      legal_mask_p2 = carry.legal_actions[1].astype(bool)
      for a1i, a1 in enumerate(legal_mask_p1):
        for a2i, a2 in enumerate(legal_mask_p2):
          if a1 <= 0.5 or a2 <= 0.5:
            continue
          new_carries = get_new_carries(carry, jnp.asarray([a1i, a2i]), turn)
          for new_carry in new_carries:
            #print("Appending new carry: ", new_carry)
            q.append((new_carry, turn + 1))
      
      visited.append(jnp.array_str(state))
    #print("State visited")
    #print(visited)
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