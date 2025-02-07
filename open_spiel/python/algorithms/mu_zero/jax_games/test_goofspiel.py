import pyspiel
import argparse
import chex
import jax
import jax.numpy as jnp
from jax_goofspiel import JaxOriginalGoofspiel, GoofspielGameState
from game_test_utils import extract_from_spiel ,histogram, compare_hists
from collections import deque

parser = argparse.ArgumentParser()

parser.add_argument("--cards", type=int, default=3, help= "Number of goofspiel cards")
parser.add_argument("--points_order", type=str, default="descending", help= "Goofspiel point card order.")

@chex.dataclass(frozen=True)
class SampleTrajectoryCarry:
  game_state: GoofspielGameState
  terminal: chex.Array
  key: chex.Array
  legal_actions: chex.Array

  

#Extract from jax goofspiel how
#many states fall under each infoset/public state
def extract_from_jax_goofspiel(cards, points_order):
  game = JaxOriginalGoofspiel(cards=cards, points_order=points_order)
  #state_tensors = []
  count_states_by_isets = [{} for _ in range(2)]
  count_states_by_public_state = {}
  all_actions = jnp.tile(jnp.arange(cards), (2, 1))
  #this is not a game with chance nodes, key will not be used
  key = jax.random.key(42)
  keys = jax.random.split(key, 1)
  #represent state as the carry information from it
  vectorized_init = jax.vmap(game.initialize_structures, in_axes=(0), out_axes=(0, 0, 0))
  game_state, key, legals = vectorized_init(keys)
  vectorized_get_info = jax.vmap(game.get_info, in_axes=(0), out_axes=(0, 0, 0, 0))
  vectorized_apply_action = jax.vmap(game.apply_action, in_axes=(0, 0, None, 0), out_axes=(0, 0, 0, 0, 0))
  def get_new_carry(carry : SampleTrajectoryCarry, joint_action, turn):
    game_state, key, terminal, rewards, legals = vectorized_apply_action(carry.game_state, carry.key, turn, joint_action)
    new_carry = SampleTrajectoryCarry(
      game_state = game_state,
      terminal= terminal,
      key = key,
      legal_actions = legals
    )
    return new_carry
  q = deque()
  visited = []
  init_carry = SampleTrajectoryCarry(
                        game_state = game_state,
                        terminal= jnp.zeros([1,1], dtype=bool),
                        key = key,
                        legal_actions = legals)
  q.append((init_carry, 0))
  while len(q) > 0:
    carry, turn = q.popleft()
    state, p1_iset, p2_iset, public_state = vectorized_get_info(carry.game_state)
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
          new_carry = get_new_carry(carry, jnp.asarray([a1, a2])[jnp.newaxis, ...], turn)
          if not new_carry.terminal:
            q.append((new_carry, turn + 1))
  return count_states_by_isets[0].values(), count_states_by_isets[1].values(), count_states_by_public_state.values()



def main():
  args= parser.parse_args()
  params = {"num_cards" : args.cards, "imp_info" : True, "points_order" : args.points_order, "num_turns" : args.cards}
  game = pyspiel.load_game("goofspiel", params)
  p1_iset_groups, p2_iset_groups, public_states_groups = extract_from_spiel(game, provides_public_state=True)
  jax_p1_iset_groups, jax_p2_iset_groups, jax_public_states_groups = extract_from_jax_goofspiel(args.cards, args.points_order)
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
  public_state_hist = histogram(list(public_states_groups))
  jax_public_state_hist = histogram(list(jax_public_states_groups))
  print("PUBLIC STATES:")
  print("SPIEL:")
  print(public_state_hist)
  print("JAX")
  print(jax_public_state_hist)
  compare_hists(public_state_hist, jax_public_state_hist)

if __name__ == "__main__":
  main()