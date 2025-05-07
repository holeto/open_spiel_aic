import numpy as np
import jax
import chex
import jax.numpy as jnp

import functools
from open_spiel.python.algorithms.mu_zero.jax_games.jax_game import JaxGame, GameState

@chex.dataclass(frozen=True)
class OshiZumoGameState(GameState):
  wrestler_position: int # Position of the wrestler on the board
  p1_coins: int  # Number of coins player 1 has
  p2_coins: int  # Number of coins player 2 has
  p1_bets: chex.Array  # History of player 1's bets
  p2_bets: chex.Array  # History of player 2's bets

class JaxOshiZumo(JaxGame):
  def __init__(self, board_size: int, initial_coins: int, max_turns: int = -1) -> None:
    """
    Initialize Oshi Zumo game.
    
    Args:
        board_size: Size of the board (2K)
        initial_coins: Number of coins each player starts with
        max_turns: Maximum number of turns (-1 for unlimited)
    """
    assert board_size % 2 == 1, "Board size must be odd, so the wrestler can be in the middle" 
    self.board_size = board_size
    self.initial_coins = initial_coins
    self.max_turns = max_turns if max_turns > 0 else initial_coins
    self.max_reward = board_size // 2
      
      

  def max_trajectory_length(self):
    return self.max_turns

  def num_distinct_actions(self):
    return self.initial_coins + 1  # Can bet 0 to initial_coins

  def information_state_tensor_shape(self):
    # Player + Position + Player Coins + Player Bet history + Turn results
    return 2 + self.board_size + self.num_distinct_actions() + self.num_distinct_actions() * self.max_turns + 2 * self.max_turns

  def public_state_tensor_shape(self):
    #
    return self.board_size + self.num_distinct_actions() * self.max_turns + 2 * self.max_turns

  @functools.partial(jax.jit, static_argnums=(0,))
  def initialize_structures(self, key): 
    game_state = OshiZumoGameState(
        wrestler_position=self.board_size // 2,
        p1_coins=self.initial_coins,
        p2_coins=self.initial_coins,
        p1_bets=jnp.full((self.max_turns, ), -1),
        p2_bets=jnp.full((self.max_turns, ), -1)
    )
    
    # Legal actions are all possible bets (0 to remaining coins)
    legal_actions = jnp.ones((2, self.initial_coins + 1))
    
    return game_state, legal_actions


  @functools.partial(jax.jit, static_argnums=(0,))
  def get_info(self, game_state: OshiZumoGameState):
    # Create state tensor with position, coins, and bet history
    
    wrestler_position = jax.nn.one_hot(game_state.wrestler_position, self.board_size)
    p1_coins = jax.nn.one_hot(game_state.p1_coins, self.initial_coins + 1)
    p2_coins = jax.nn.one_hot(game_state.p2_coins, self.initial_coins + 1)
    p1_bets = jax.nn.one_hot(game_state.p1_bets, self.initial_coins + 1).flatten() 
    p2_bets = jax.nn.one_hot(game_state.p2_bets, self.initial_coins + 1).flatten() 
    
    p1_wins = jnp.where(game_state.p1_bets > game_state.p2_bets, 1, 0)
    p2_wins = jnp.where(game_state.p2_bets > game_state.p1_bets, 1, 0)
    
    state_tensor = jnp.concatenate([
        wrestler_position,
        p1_coins,
        p2_coins,
        p1_bets,
        p2_bets
    ])
    
    player_id = jax.nn.one_hot(0, 2)
    p2_bets
    p1_iset_tensor = jnp.concatenate([
        player_id,
        wrestler_position,
        p1_coins,
        p1_bets,
        p1_wins,
        p2_wins,
    ]) 
    
    # We swap the winning sequence for the player 2, just so it is more symmetric for the networks.
    p2_iset_tensor = jnp.concatenate([
        1 - player_id,
        wrestler_position,
        p2_coins,
        p2_bets,
        p2_wins,
        p1_wins,
    ]) 
    
    
    tie_card = jnp.where(jnp.logical_and(game_state.p1_bets == game_state.p2_bets, game_state.p1_bets >= 0), game_state.p1_bets, -1)
    tie_card_oh = jax.nn.one_hot(tie_card, self.initial_coins + 1).flatten()
    # For public state we also store the tie card, because here we do not p2_betshave actions of each player to distinguish between different ties.
    public_state_tensor = jnp.concatenate([
        wrestler_position,
        p1_wins,
        p2_wins,
        tie_card_oh,
    ])
    return state_tensor, p1_iset_tensor, p2_iset_tensor, public_state_tensor

  @functools.partial(jax.jit, static_argnums=(0,))
  def apply_action(self, game_state: OshiZumoGameState, key, turn, actions):
    # p1_bet, p2_bet = actions
    p1_bet, p2_bet = actions[0], actions[1]
    # Update bet historyOshiZumoGame
    new_p1_bets = game_state.p1_bets.at[turn].set(p1_bet)
    new_p2_bets = game_state.p2_bets.at[turn].set(p2_bet)
    
    # Update coins
    new_p1_coins = game_state.p1_coins - p1_bet
    new_p2_coins = game_state.p2_coins - p2_bet
    
    # Either you move for +1 if player 1 has higher bet or -1 if opponent has higher bet.
    new_wrestler_position = game_state.wrestler_position + (p1_bet > p2_bet) - (p2_bet > p1_bet)
    
    new_game_state = OshiZumoGameState(
        wrestler_position=new_wrestler_position,
        p1_coins=new_p1_coins,
        p2_coins=new_p2_coins,
        p1_bets=new_p1_bets,
        p2_bets=new_p2_bets
    )
    # legal_actions = jnp.zeros((2, self.initial_coins + 1))

    new_p1_coins_oh = jax.nn.one_hot(new_p1_coins + 1, self.initial_coins + 1)
    new_p2_coins_oh = jax.nn.one_hot(new_p2_coins + 1, self.initial_coins + 1)
    new_coins_oh = jnp.stack([new_p1_coins_oh, new_p2_coins_oh], 0)
    legal_actions = 1 - jnp.cumsum(new_coins_oh, -1)
    
    # legal_actions = legal_actions.at[0, :new_p1_coins+1].set(1)
    # legal_actions = legal_actions.at[1, :new_p2_coins+1].set(1)
    
    wrestler_at_edge = jnp.where(jnp.logical_or(new_wrestler_position == 0, new_wrestler_position == self.board_size - 1), 1, 0)
    # no_more_coins = jnp.where(new_p1_coins == 0 or new_p2_coins== 0, 1, 0)
    no_more_turns = jnp.where(turn == self.max_turns - 1, 1, 0)
    no_more_coins = jnp.where(jnp.logical_and(new_p1_coins == 0, new_p2_coins == 0), 1, 0)
    
    terminal = jnp.logical_or(wrestler_at_edge, no_more_turns)
    terminal = jnp.logical_or(terminal, no_more_coins)
    
    # This is reward based on the poition on the grid
    # We scale it to [-1; 1]
    positional_reward = (new_wrestler_position - self.board_size // 2) / self.max_reward
    
    reward = jnp.where(terminal, positional_reward, 0)
    
    
    return new_game_state, terminal, reward, legal_actions
  
if __name__ == "__main__":
  game = JaxOshiZumo(7, 10)
  game_state, legal_actions = game.initialize_structures(jax.random.PRNGKey(0))
  state_tensor, p1_iset_tensor, p2_iset_tensor, public_state_tensor = game.get_info(game_state)
  # print(state_tensor)
  print(p1_iset_tensor.shape)
  # print(p2_iset_tensor)
  print(public_state_tensor.shape)
  print(game.information_state_tensor_shape())
  print(game.public_state_tensor_shape())
  
  new_game_state, terminal, rewards, new_legal_actions = game.apply_action(game_state, jax.random.PRNGKey(0), 0, jnp.array([1, 1]))
  # state_tensor, p1_iset_tensor, p2_iset_tensor, public_state_tensor = game.get_info(new_game_state)
  
  # print(state_tensor)
  # print(p1_iset_tensor)
  # print(p2_iset_tensor)
  # print(public_state_tensor)
  
  
  new_game_state, terminal, rewards, new_legal_actions = game.apply_action(new_game_state, jax.random.PRNGKey(0), 1, jnp.array([2, 4]))
  # state_tensor, p1_iset_tensor, p2_iset_tensor, public_state_tensor = game.get_info(new_game_state)
  
  # print(state_tensor)
  # print(p1_iset_tensor)
  # print(p2_iset_tensor)
  # print(public_state_tensor)
  

  new_game_state, terminal, rewards, new_legal_actions = game.apply_action(new_game_state, jax.random.PRNGKey(0), 2, jnp.array([3, 4])) 
  
  # print(new_game_state)
  # print(state_tensor)
  # print(p1_iset_tensor)
  # print(p2_iset_tensor)
  # print(public_state_tensor)