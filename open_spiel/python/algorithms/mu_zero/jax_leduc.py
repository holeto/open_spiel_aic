import jax
import jax.numpy as jnp

import functools

INVALID_ID = 0
FOLD_ID = 1
CALL_ID = 2
RAISE_ID = 3


class JaxOriginalLeduc:
  def __init__(self, starting_player=0):
    #Invalid action, fold, call, raise
    self.num_actions = 4
    #Two rounds and in each the maximum length trajectory consists of
    # actions Call, Raise, Raise, [Call or Fold], minus one action
    # because terminals are not counted
    self.max_turns = 7
    #Cards in two suits, three cards from each suit
    self.total_cards = 6
    self.players = 2
    self.starting_player = starting_player
    #Max 4 raises. Starting at 1, raises in the first
    # round are 2 + 2 and in the second round 4 + 4
    self.max_bet_amount = 13
    self.max_raises_per_round = 2
    self.raise_amount = 2
    #assuming that cards of different suits are counted as 
    #distinct cards
    self.private_chance_outcomes = 30
    #there are 4 cards left in the deck
    #self.public_chance_outcomes = 4
    #self.chance_outcomes = 120
    #JAX constants TODO: Probably put this somewhere else
    self.invalid_action_mask = jax.nn.one_hot(INVALID_ID, self.num_actions)
    #self.possible_public_cards = None
    #self.player_card_types = None


  def new_initial_state(self):
    #TODO: Complete this
    return 0
  
  def num_distinct_actions(self):
    return self.num_actions
  
  def information_state_tensor_shape(self):
    # One hot encoded receiving player
    # One hot encoded private card of player
    # One hot encoded public card (1 bit added to recognize not yet revealed)
    # One hot encoded actions in each turn
    return self.players + self.total_cards + self.total_cards + 1  + self.max_turns * (self.num_actions - 1)
  
  def public_state_tensor_shape(self):
    # One hot encoded public card (1 bit added to recognize not yet revealed)
    # One hot encoded actions in each turn (not the invalid added actions)
    return self.total_cards + 1 + self.max_turns * (self.num_actions - 1)
  
  @functools.partial(jax.jit, static_argnums=(0))
  def _jit_initialize_structures(self, key):
    init_reaches = jnp.ones(self.private_chance_outcomes) / self.private_chance_outcomes
    fold_oh = jax.nn.one_hot(FOLD_ID, self.num_actions)
    starting_action_mask = jnp.ones(self.num_actions) - self.invalid_action_mask - fold_oh
    if self.starting_player == 0:
      p1_legal_mask = starting_action_mask
      p2_legal_mask = self.invalid_action_mask
    if self.starting_player == 1:
      p1_legal_mask = self.invalid_action_mask
      p2_legal_mask = starting_action_mask
    current_chips = jnp.ones(self.players)
    #[H, A]
    action_history = jnp.zeros([self.max_turns, self.num_actions - 1])
    cards = jnp.arange(self.total_cards)[..., None]
    p1_private_cards = jnp.repeat(cards, self.total_cards - 1, axis=0)
    #TODO: This can probably be done better.
    p2_private_cards = jnp.concatenate([jnp.r_[0:i:1, i+1:self.total_cards:1] for i in range(self.total_cards)])[..., None]
    private_cards = jnp.concatenate([p1_private_cards, p2_private_cards], axis=1)
    #private_cards = jnp.repeat(jnp.concatenate([p1_private_cards, p2_private_cards], axis=1), [self.public_chance_outcomes, 1])
    public_card = jnp.zeros(1, dtype=int)
    round = jnp.zeros(1, dtype=int)
    turns_this_round = jnp.zeros(1, dtype=int)
    chosen_cards = jax.random.choice(key, private_cards, axis=0)
    key = jax.random.split(key, 1)[0]
    legals = jnp.stack([p1_legal_mask, p2_legal_mask], axis=0)
    return action_history, public_card, chosen_cards, current_chips, key, round, turns_this_round, init_reaches, legals
  
  @functools.partial(jax.jit, static_argnums=(0, 1))
  def _jit_batch_initialize_structures(self, batch, key):
    init_reaches = jnp.ones(self.private_chance_outcomes) / self.private_chance_outcomes
    fold_oh = jax.nn.one_hot(FOLD_ID, self.num_actions)
    starting_action_mask = jnp.ones(self.num_actions) - self.invalid_action_mask - fold_oh
    if self.starting_player == 0:
      p1_legal_mask = starting_action_mask
      p2_legal_mask = self.invalid_action_mask
    if self.starting_player == 1:
      p1_legal_mask = self.invalid_action_mask
      p2_legal_mask = starting_action_mask
    current_chips = jnp.ones([batch, self.players])
    #[Batch, H, A]
    action_history = jnp.zeros([batch, self.max_turns, self.num_actions - 1])
    cards = jnp.arange(self.total_cards)[..., None]
    p1_private_cards = jnp.repeat(cards, self.total_cards - 1, axis=0)
    #TODO: This can probably be done better.
    p2_private_cards = jnp.concatenate([jnp.r_[0:i:1, i+1:self.total_cards:1] for i in range(self.total_cards)])[..., None]
    private_cards = jnp.concatenate([p1_private_cards, p2_private_cards], axis=1)
    chosen_cards = jax.random.choice(key, private_cards, axis=0, shape=(batch, ))
    #private_cards = jnp.repeat(jnp.concatenate([p1_private_cards, p2_private_cards], axis=1), [self.public_chance_outcomes, 1])
    public_card = jnp.zeros([batch, 1], dtype=int)
    round = jnp.zeros([batch, 1], dtype=int)
    turns_this_round = jnp.zeros([batch, 1], dtype=int)
    keys = jax.random.split(key, batch)
    #[Batch, Pl, A]
    batch_legals = jnp.tile(jnp.stack([p1_legal_mask, p2_legal_mask], axis=0), (batch, 1, 1))
    return action_history, public_card, chosen_cards, current_chips, keys, round, turns_this_round, init_reaches, batch_legals

  def initialize_structures(self, key, batch=-1):
    if batch ==-1:
      action_history, public_card, chosen_private_cards, current_chips, keys, round, turns_this_round, init_reaches, legals = self._jit_initialize_structures(key)
    else:
      action_history, public_card, chosen_private_cards, current_chips, keys, round, turns_this_round, init_reaches, legals = self._jit_batch_initialize_structures(batch, key)   
    #used_cards = jnp.sort(chosen_private_cards.ravel())
    #self.possible_public_cards = jnp.r_[0:used_cards[0]:1, used_cards[0]+1:used_cards[1]:1, used_cards[1]+1:self.total_cards:1]
    #Integer division by 2 places cards into the [J1, J2], [Q1, Q2], [K1, K2] buckets
    #self.player_card_types = jnp.floor_divide(used_cards, 2)
    return action_history, public_card,  chosen_private_cards, current_chips, keys, round, turns_this_round, init_reaches, legals
  
  @functools.partial(jax.jit, static_argnums=(0))
  def get_info(self, action_history, public_card, private_cards):
    #One additional bit for public card not dealt yet
    public_card_oh = jax.nn.one_hot(public_card, self.total_cards + 1)
    public_state_tensor = jnp.concatenate([public_card_oh.ravel(), action_history.ravel()], axis=0)
    private_cards_oh = jax.nn.one_hot(private_cards, self.total_cards)

    p1_player = jax.nn.one_hot(0, 2)
    
    p1_iset_tensor = jnp.concatenate([p1_player.ravel(), private_cards_oh[0], public_state_tensor], axis=0)
    p2_iset_tensor = jnp.concatenate([1 - p1_player.ravel(), private_cards_oh[1], public_state_tensor], axis=0)

    state_tensor = jnp.concatenate([private_cards_oh.ravel(), public_state_tensor], axis=0)

    return state_tensor, p1_iset_tensor, p2_iset_tensor, public_state_tensor
  

  #TODO: This could surely be improved by passing and returning less stuff
  @functools.partial(jax.jit, static_argnums=(0))
  def apply_action(self, action_history, public_card, private_cards, current_chips, key, round, turns_this_round, turn, actions):
    oh_actions = jax.nn.one_hot(actions, self.num_actions)
    oh_turn = jax.nn.one_hot(turn, self.max_turns)
    fold_oh = jax.nn.one_hot(FOLD_ID, self.num_actions)
    raise_oh = jax.nn.one_hot(RAISE_ID, self.num_actions)

    next_player = (self.starting_player + ((turn + 1) % 2)) % self.players
    max_chips = jnp.max(current_chips)
    oh_valid_action = jax.nn.one_hot(actions[1- next_player] - 1, self.num_actions - 1)

    #Integer division by 2 places cards into the [J1, J2], [Q1, Q2], [K1, K2] buckets
    player_card_types = jnp.floor_divide(private_cards, 2)
    possible_public_card_mask = 1 - jnp.sum(jax.nn.one_hot(private_cards, self.total_cards), axis=0)
    possible_public_cards = jnp.flatnonzero(possible_public_card_mask, size=self.total_cards - 2)

    
    folded = jnp.any(oh_actions[1 - next_player] * fold_oh)
    raised = jnp.sum(oh_actions * raise_oh, axis=1)

    tie = jnp.all(jnp.isclose(player_card_types[0], player_card_types[1]))
    card_matched = jnp.any(jnp.floor_divide(public_card - 1, 2) == player_card_types)
    winner = jnp.where(card_matched, jnp.argmin(jnp.abs(jnp.floor_divide(public_card - 1, 2) - player_card_types)), jnp.argmax(player_card_types))
    winner = jnp.where(folded, next_player, winner)

    this_turn_played = oh_valid_action[..., None, :] * oh_turn[None, :, None]
    action_history = (action_history + this_turn_played)[0]

    #Taking advantage of the fact, that raises can only happen after each other
    num_raises = jnp.where(turn > 0, action_history[turn - 1, RAISE_ID - 1] + raised[1 - next_player], 0)

    action_chips = jnp.concatenate([jnp.repeat(current_chips[..., None], 2, axis =1), jnp.array([max_chips, max_chips])[..., None] * jnp.ones(2)], axis=1)
    current_chips = jnp.sum(action_chips * oh_actions, axis=1)
    current_chips = current_chips + (raised * (round + 1) * self.raise_amount)
    
    bets_equal = jnp.all(jnp.isclose(current_chips[0], current_chips[1]))
    
    new_acting_legals = jnp.ones(self.num_actions) - self.invalid_action_mask
    #whether raise is still possible
    new_acting_legals = jnp.where(num_raises < self.max_raises_per_round, new_acting_legals, new_acting_legals - raise_oh)
    #whether fold is possible
    new_acting_legals = jnp.where(bets_equal, new_acting_legals - fold_oh, new_acting_legals)
    new_legals = jnp.where(next_player == 0, jnp.stack([new_acting_legals, self.invalid_action_mask], axis=0), jnp.stack([self.invalid_action_mask, new_acting_legals], axis=0))

    play_chance = jnp.logical_and(jnp.logical_and(turn >= 1, round == 0), bets_equal)
    public_card = jnp.where(play_chance, jax.random.choice(key, possible_public_cards) + 1, public_card)
    round = round + play_chance
    #make sure to properly reset to ready for the new round
    turns_this_round = jnp.where(play_chance, -1, turns_this_round)

    terminal = jnp.logical_or(folded, jnp.logical_and(jnp.logical_and(round > 0, turns_this_round >= 1), bets_equal))
    
  
    #The division by self.max_bet_amount to make sure the rewards is normalized to [-1, 1] range
    reward = jnp.where(terminal, jnp.where(jnp.logical_and(tie, ~folded), 0, ((1 - 2 * winner) * current_chips[1-winner]) / self.max_bet_amount), 0)
    
    key = jax.random.split(key, 1)[0]

    return action_history, public_card, private_cards, current_chips, key, round, turns_this_round + 1, terminal, jnp.asarray([reward, -reward]), new_legals