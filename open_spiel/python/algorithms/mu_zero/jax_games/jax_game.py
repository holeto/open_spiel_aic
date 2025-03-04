from abc import ABC, abstractmethod
import chex


class JaxPolicy:
  policy: dict[str, list[float]]
  
  def __init__(self, policy: dict[str, list[float]] = None) -> None:
    self.policy = {}
    # Using default {} broke things
    if policy is not None:
      self.policy = policy
  
  def __getitem__(self, key: str) -> list[float]:
    return self.policy[key]
  
  def __setitem__(self, key: str, value: list[float]) -> None:
    self.policy[key] = value
    
    
class GameState(ABC):
  pass


#Abstract parent class for any jax game
class JaxGame(ABC):
  
  def new_initial_state(self, key):
    init_state, legals = self.initialize_structures(key)
    return self.get_info(init_state)
  
  @abstractmethod
  def num_distinct_actions(self):
    pass
  
  #returns game_state, key, legal_actions
  @abstractmethod
  def initialize_structures(self, key):
    pass

  #returns state_tensor, p1_iset_tensor, p2_iset_tensor, public_state_tensor
  @abstractmethod
  def get_info(self, game_state):
    pass

  #returns new_game_state, key, terminal, rewards, new_legals
  @abstractmethod
  def apply_action(self, game_state, key, turn, actions):
    pass