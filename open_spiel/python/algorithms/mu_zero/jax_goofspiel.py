import numpy as np
import jax
import jax.numpy as jnp

import functools


class JaxOriginalGoofspiel:
  def __init__(self, cards, points_order="descending", turns=-1) -> None:
    self.cards = cards
    self.turns = turns
    if turns <= 0:
      self.turns = cards
    self.points_order = points_order 
  
  def new_initial_state(self):
    return 0
  
  def num_distinct_actions(self):
    return self.cards
  
  def information_state_tensor_shape(self):
    return self.turns * self.cards + self.turns * 2 + self.turns * self.cards * 2 + 2
  
  def public_state_tensor_shape(self):
    return self.turns * self.cards + self.turns * 2 + self.turns * self.cards
  
  def initialize_structures(self):
    if self.points_order == "descending":
      point_cards = np.arange(self.cards, self.cards - self.turns, -1)
    if self.points_order == "ascending":
      point_cards = np.arange(1, 1 + self.cards - self.turns)
    played_cards = np.zeros((2, self.turns, self.cards))
    p1_points = np.zeros(self.turns)
    return point_cards, played_cards, p1_points, np.ones((2, self.cards))
  
  @functools.partial(jax.jit, static_argnums=(0, 1))
  def initialize_batch_structures(self, batch):
    if self.points_order == "descending":
      point_cards = jnp.tile(jnp.arange(self.cards, self.cards - self.turns, -1), (batch, 1))
    if self.points_order == "ascending":
      point_cards = jnp.tile(jnp.arange(1, 1 + self.cards - self.turns), (batch, 1))
    played_cards = jnp.zeros((batch, 2, self.turns, self.cards))
    p1_points = jnp.zeros((batch, self.turns))
    return point_cards, played_cards, p1_points, jnp.ones((batch, 2, self.cards))
  
  # State Tensor -> Point card [Turn, Card], Winner [Turn, Player], Tie Cards [Turn, Card], Played Cards [Player, Turn, Card], 
  # Iset tensor -> Observing Player, Point card [Turn, Card], Winner [Turn, Player], Tie Cards [Turn, Card], Played Cards [Turn, Card],
  # Public tensor -> Point card [Turn, Card], Winner [Turn, Player], Tie Cards [Turn, Card]
  @functools.partial(jax.jit, static_argnums=(0,))
  def get_info(self, point_cards, played_cards, p1_points):
    played_turns_mask = jnp.sum(played_cards[0], -1)
    # To set the first to 
    played_turns_mask = jnp.roll(played_turns_mask, 1, axis=0) + jax.nn.one_hot(0, self.turns)
    # Every card that is played have value >= 1, non-played has 0. So we just subtract 1 to make sure everything works with one-hot (-1 is all zeros)
    point_cards_masked = point_cards * played_turns_mask - 1  
    oh_point_cards = jax.nn.one_hot(point_cards_masked, self.cards)
    
    # Tie -1, P1 win 0, P2 win 1
    p2_winned = jnp.where(p1_points < 0, 1, 0) - (p1_points == 0)
    winner = jax.nn.one_hot(p2_winned, 2)
    
    tie_cards = jnp.expand_dims(((p1_points == 0) * played_turns_mask), -1) * played_cards[0]
    
    public_state_tensor = jnp.concatenate([jnp.ravel(oh_point_cards), jnp.ravel(winner), jnp.ravel(tie_cards)], axis=0)
    
    p1_player = jax.nn.one_hot(0, 2)
    
    p1_iset_tensor = jnp.concatenate([p1_player, public_state_tensor, jnp.ravel(played_cards[0])], axis=0)
    p2_iset_tensor = jnp.concatenate([1 - p1_player, public_state_tensor, jnp.ravel(played_cards[1])], axis=0)
    
    state_tensor = jnp.concatenate([public_state_tensor, jnp.ravel(played_cards)], axis=0)
    
    
    return state_tensor, p1_iset_tensor, p2_iset_tensor, public_state_tensor
  
  @functools.partial(jax.jit, static_argnums=(0,))
  def apply_action(self, point_cards, played_cards, p1_points, turn, actions):
    # This is not working
    # turn = jnp.argmax(jnp.arange(self.turns) * jnp.sum(played_cards[0], -1))
    oh_actions = jax.nn.one_hot(actions, self.cards) 
    oh_turn = jax.nn.one_hot(turn, self.turns)
    
    winner = jnp.argmax(actions, axis=-1)
    loser = jnp.argmin(actions, axis=-1)
    tie = winner == loser
    
    # Point cards are from 0 to N-1, but points should be from 1 to N
    point = point_cards[..., turn] * oh_turn
    
    this_turn_played = oh_actions[..., None, :] * oh_turn[None, :, None]
    
    played_cards = played_cards + this_turn_played
    
    p1_points = jnp.where(tie, p1_points, jnp.where(winner == 0, p1_points + point, p1_points - point))
    
    legal_actions = 1 - jnp.sum(played_cards, 1)
    
    next_action = jnp.argmax(legal_actions, -1)
    next_winner = jnp.argmax(next_action)
    next_loser = jnp.argmin(next_action)
    next_tie = next_winner == next_loser
    next_point = point_cards[..., turn+1] * jax.nn.one_hot(turn+1, self.turns)
    
    p1_points = jnp.where(turn != self.cards - 2, p1_points, jnp.where(next_tie, p1_points, jnp.where(next_winner == 0, p1_points + next_point, p1_points - next_point)))
    
    rewards = jnp.where(turn != self.cards - 2, 0, jnp.clip(jnp.sum(p1_points), -1, 1) )
    
    # rewards = jnp.sum(p1_points)
    # if turn == self.cards-1:
    #   actions = jnp.argmax(legal_actions, -1)
    #   return self.apply_action(point_cards, played_cards, p1_points, turn+1, actions)
    
    return legal_actions, rewards, point_cards, played_cards, p1_points
    


class JaxGoofspiel():
  def __init__(self, cards, turns, first_card) -> None:
    
    self.cards = cards
    self.turns = turns
    self.first_card = first_card

  # TODO(kubicon): Test whether np or jnp is better for this usecase
  def initialize_structures(self):
    point_cards = np.zeros((self.turns, self.cards))
    played_cards = np.zeros((2, self.turns, self.cards))
    p1_points = np.zeros(self.turns)
    # current_point_card = self.first_card
    point_cards[0, self.first_card] = 1
    return point_cards, played_cards, p1_points
    
  def get_tensor(self, point_cards, played_cards, p1_points):
    
    pass
  
  # Input dimensions are [..., T, C], [..., 2, T, C], [..., T], [...], [..., 2]
  # @functools.partial(jax.jit, static_argnums=(0,))
  def apply_action(self, point_cards, played_cards, p1_points, turn, actions):
    oh_actions = jax.nn.one_hot(actions, self.cards) # [..., 2, C]
    oh_turn = jax.nn.one_hot(turn, self.turns) # [..., T]
  
    winner = jnp.argmax(actions, axis=-1) # [...]
    loser = jnp.argmin(actions, axis=-1) # [...]
    
    #TODO(kubicon): This expects that argmin and argmax return first max and first min. This may not be the case in jitted version, so comparing max and min may be better
    tie = winner == loser # Max and min are the same, so its the tie. 
    
    oh_point = oh_turn * jnp.argmax(point_cards[turn], axis=-1) #[..., T]
    # If Tie, copy the points, if not add or subtract the points
    p1_points = jnp.where(tie, p1_points, jnp.where(winner == 0, p1_points + oh_point, 
                                                    p1_points - oh_point)) # [..., T] 
    
    # Invalidate played action
    played_cards = played_cards + jax.nn.one_hot(oh_turn * actions)
    # player_cards = jnp.where(oh_actions, 0, player_cards) 
    
    turn = turn + 1
    
    # In turn=0, and cards=13, we want to set 2nd round card to be 11, so even if we added 1 to turn, we have to add one more
    next_point_card_value = self.turns - (turn + 1) # [...]
    # You could also do it by shiftin oh_turn -> Check which is faster
    oh_next_turn = jax.nn.one_hot(turn, self.turns) # [..., T]
    next_point_card = oh_next_turn * next_point_card_value # [..., T]
    oh_next_point = jax.nn.one_hot(next_point_card, self.cards) #[..., T, C]
    #
    point_cards = point_cards + oh_next_point
    # point_cards = jnp.where(oh_next_point, 1, point_cards)
    
    return point_cards, played_cards, p1_points, turn
  
from pyinstrument import Profiler

def main():
  cards = 5
  batch = 32
  goof = JaxOriginalGoofspiel(cards)
  info, legal_actions = goof.initialize_batch_structures(batch)
  
  apply_action = jax.vmap(goof.apply_action, in_axes=(0, 0, 0, None, 0), out_axes=(0, 0, 0))
  rng = jax.random.PRNGKey(0)
  get_info = jax.vmap(goof.get_info, in_axes=(0, 0, 0), out_axes=(0, 0, 0, 0))
  
  @jax.jit
  def choice_wrapper(key, p):
    return jax.random.choice(key, cards, p=p)


  sample_action = jax.vmap(jax.vmap(choice_wrapper, in_axes=(0, 0), out_axes=0), in_axes=(0, 0), out_axes=0)
  
  @jax.jit
  def do_smth(temp_key, legal_actions, info):
    keys = jax.random.split(temp_key, (batch, 2))
    actions = sample_action(keys, legal_actions)
    legal_actions, rewards, info = apply_action(*info, i, actions)
    neco = get_info(*info)
    return legal_actions, rewards, neco, info


  a = Profiler()
  a.start()
  for j in range(10000):
    info, legal_actions = goof.initialize_batch_structures(batch)
    for i in range(4):
      rng, temp_key = jax.random.split(rng)
      legal_actions, rewards, neco, info = do_smth(temp_key, legal_actions, info)
  a.stop()
  a.print()
    
  
  
if __name__ == "__main__":
  main()