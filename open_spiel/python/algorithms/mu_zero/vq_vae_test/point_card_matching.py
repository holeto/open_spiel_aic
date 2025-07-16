import jax
import jax.numpy as jnp

import chex
import functools

from open_spiel.python.algorithms.mu_zero.jax_games.jax_game import JaxGame, GameState

"""A single player goofspiel inspired game.
To mantain consistency with other JAX games,
it actually is treated as a simultaneous move
game, but the second player always has only
one legal action.
Basically just a match the point card game.
Just a sanity check for the VQ-VAE algorithm"""

@chex.dataclass(frozen=True)
class PointCardMatchingState(GameState):
  #history of one hot played cards
  played_cards: chex.Array
  #history of one hot point cards
  points: chex.Array
  point_cards: chex.Array
  terminal: chex.Array


class PointCardMatching(JaxGame):
  def __init__(self, num_cards):
    self.num_cards = num_cards
    self.max_turns = num_cards
    #get a reward 1 whenever a card is matched
    self.max_points = num_cards

  

  def initialize_structures(self, key):
    init_played_cards = jnp.zeros((self.max_turns, self.num_cards))
    init_points = jnp.zeros(1)
    init_point_cards = jnp.concatenate([jax.nn.one_hot(self.num_cards - 1, self.num_cards)[None, ...], jnp.zeros((self.max_turns - 1, self.num_cards))], axis=0)
    init_state = PointCardMatchingState(played_cards = init_played_cards,
                                        points = init_points,
                                        point_cards = init_point_cards,
                                        terminal = jnp.array(False))
    init_legals = jnp.stack([jnp.ones(self.num_cards), jax.nn.one_hot(0, self.num_cards)], axis=0)
    return init_state, init_legals
  
  @functools.partial(jax.jit, static_argnums=(0))
  def get_info(self, state: PointCardMatchingState):
    #starting at 0 points hence the + 1
    points_oh = jax.nn.one_hot(state.points, self.max_points + 1)
    state_tensor = jnp.concatenate([state.played_cards.ravel(), state.point_cards.ravel(), points_oh.ravel()])
    #Return just a state tensor
    # this is a perfect information game
    return state_tensor, state_tensor, state_tensor, state_tensor
  
  def num_distinct_actions(self):
    return self.num_cards
  
  def state_tensor_size(self):
    return 2 * (self.max_turns * self.num_cards) + self.max_points + 1
  
  @functools.partial(jax.jit, static_argnums=(0))
  def apply_action(self, state: PointCardMatchingState, key, turn, actions):
    turn_oh = jax.nn.one_hot(turn, self.max_turns)
    point_card_turn_oh = jax.nn.one_hot(turn + 1, self.max_turns)
    action = actions[0]
    action_oh = jax.nn.one_hot(action, self.num_cards)

    new_played_cards = state.played_cards + (action_oh[None, ...] * turn_oh[..., None])
    already_played = jnp.sum(new_played_cards, axis=0)
    new_legals = jnp.ones(self.num_cards) - already_played
    new_legals = jnp.stack([new_legals, jax.nn.one_hot(0, self.num_cards)], axis=0)

    #descending order
    point_card = self.max_turns - turn - 2
    new_points = state.points + (point_card == action)

    point_card_oh = jax.nn.one_hot(point_card, self.num_cards)
    new_point_cards = state.point_cards + (point_card_oh[None, ...] * point_card_turn_oh[..., None])

    terminal = turn == (self.max_turns - 2)
    terminal = state.terminal + terminal
    #new_point_cards = jnp.where(turn ==(self.max_turns - 1), state.point_cards, new_point_cards)
    #checking if we can still match the
    #last point card, which will be the first card
    # because of descending order.
    last_card_matched = jnp.sum(new_legals * jax.nn.one_hot(0, self.num_cards))
    reward = jnp.where(terminal, new_points + last_card_matched, jnp.zeros_like(new_points + last_card_matched))

    new_state = PointCardMatchingState(played_cards=new_played_cards,
                                       point_cards = new_point_cards,
                                       points= new_points,
                                       terminal = terminal)
    return new_state, new_legals, reward, terminal