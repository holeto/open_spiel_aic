# Copyright 2019 DeepMind Technologies Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A simultaneous version of JAX-CFR, which is made to
work specifically on the jax implementation of Leduc Poker
"""

# pylint: disable=g-importing-member

from collections import namedtuple
import functools

import chex
import jax
import jax.numpy as jnp
import numpy as np

#from open_spiel.python import policy
from open_spiel.python.jax.cfr.jax_cfr import regret_matching, update_regrets, update_regrets_plus, JAX_CFR_SIMULTANEOUS_UPDATE
from open_spiel.python.jax.cfr.jax_simultaneous_cfr import SimultaneousJaxCFRConstants
from open_spiel.python.algorithms.mu_zero.jax_games.jax_leduc import JaxLeduc, LeducGameState
from open_spiel.python.algorithms.mu_zero.jax_games.jax_game import JaxPolicy
from open_spiel.python.algorithms.mu_zero.jax_games.jax_game_algorithms import stringify
import pyspiel


class JaxLeducCFR:
  """A version of SimultaneousJaxCFR specificially designed to work on
  the JAX implementation of Leduc poker. All JAX games are treated as simultaneous
  move games, so there is no checking whether to create the fake action,
  since that is assumed to be part of the game definition.

  First it prepares all the structures in `init`, then it just reuses them
  within jitted function `jit_step`.
  """

  def __init__(
      self,
      start_state_info = None,
      regret_matching_plus=True,
      alternating_updates=True,
      linear_averaging=True,
  ):
    self.game = JaxLeduc()
    self._regret_matching_plus = regret_matching_plus
    self._alternating_updates = alternating_updates
    self._linear_averaging = linear_averaging
    self.timestep = 1
    self.dummy_key = jax.random.PRNGKey(0)

    self.init(start_state_info)

  def init(self, start_state_info):
    """Constructor."""

    # This implementation will only work for 2 player games !!!
    players = 2
    #depth_history_utility = [[] for _ in range(players)]
    depth_history_utility = []
    depth_history_previous_iset = [[] for _ in range(players)]
    depth_history_previous_action = [[] for _ in range(players)]
    depth_history_iset = [[] for _ in range(players)]
    depth_history_actions = [[] for _ in range(players)]
    depth_history_next_history = []
    depth_history_player = []
    depth_history_chance = []
    depth_history_previous_history = []
    depth_history_action_mask = []
    depth_history_chance_probabilities = []
    # Previous action is mapping of both iset and action!
    iset_previous_action = [[] for _ in range(players)]
    iset_action_mask = [[] for _ in range(players)]
    iset_action_depth = [[] for _ in range(players)]
    ids = [0 for _ in range(players)]
    pl_isets = [{} for _ in range(players)]
    initial_legals = [[0, 0, 1, 1], [1, 0, 0, 0]]
    #The first chance node is not taken into 
    # the tree, so only four distinct actions should be here
    distinct_actions = self.game.num_distinct_actions()
    #distinct_joint_actions = distinct_actions ** 2

    for pl in range(players):
      pl_isets[pl][''] = ids[pl]
      ids[pl] += 1
      am = [0] * distinct_actions
      am[0] = 1
      iset_action_mask[pl].append(am)
      iset_previous_action[pl].append(0)
      iset_action_depth[pl].append(0)

    PreviousInfo = namedtuple(
        'PreviousInfo',
        ('actions', 'isets', 'prev_actions', 'history', 'turns'),
    )

    def _traverse_tree(state: LeducGameState, previous_info: PreviousInfo, legal_actions, reward, depth, is_chance_node = False, chance=1.0):

      if len(depth_history_next_history) <= depth:
        for pl in range(players):
          depth_history_previous_iset[pl].append([])
          depth_history_previous_action[pl].append([])
          depth_history_iset[pl].append([])
          depth_history_actions[pl].append([])

        depth_history_action_mask.append([])
        depth_history_next_history.append([])
        depth_history_utility.append([])
        depth_history_player.append([])
        depth_history_chance.append([])
        depth_history_previous_history.append([])
        depth_history_chance_probabilities.append([])
      history_id = len(depth_history_previous_history[depth])
      #print("State ", state)
      #breakpoint()
      #This only holds if current_player == pyspiel.PlayerId.SIMULTANEOUS.value.
      #In chance node we pretend that player 1 is acting
      acting_player = 0 if legal_actions[0][0] == 0 or is_chance_node else 1

      next_history_temp = [0] * distinct_actions
      depth_history_next_history[depth].append(next_history_temp)
      if is_chance_node:
        current_player = pyspiel.PlayerId.CHANCE.value
      elif state.terminal:
        current_player = pyspiel.PlayerId.TERMINAL.value
      else :
        current_player = acting_player
      depth_history_player[depth].append(current_player)
      depth_history_chance[depth].append(chance)
      depth_history_previous_history[depth].append(previous_info.history)
      actions_mask = [0] * distinct_actions 
      for ai, a in enumerate(legal_actions[acting_player]):
        if a < 0.5:
          continue
        actions_mask[ai] = 1
      depth_history_action_mask[depth].append(actions_mask)
      for pl in range(players):
        depth_history_previous_iset[pl][depth].append(previous_info.isets[pl])
        depth_history_previous_action[pl][depth].append(
            previous_info.actions[pl]
        )
      
      depth_history_utility[depth].append(reward)
      chance_probabilities = [0.0 for _ in range(distinct_actions)]
      if is_chance_node:
        #the inner chance nodes have 4 chance outcomes
        for a in range(4):
          chance_probabilities[a] = 0.25
      elif not state.terminal:
        chance_probabilities = [1.0 for _ in range(distinct_actions)]
      else:
        chance_probabilities = [
            1.0 / distinct_actions for _ in range(distinct_actions)
        ]
      
      depth_history_chance_probabilities[depth].append(chance_probabilities)
      if current_player >= 0:
        #acting player is the one who does not 
        # have the invalid action as legal
        #breakpoint()
        state_tensor, p1_iset, p2_iset, ps = self.game.get_info(state)
        isets = [stringify(p1_iset), ""] if acting_player == 0 else ["", stringify(p2_iset)]
        #isets=[stringify(p1_iset), stringify(p2_iset)]
        for pl in range(players):
          if pl == acting_player:
            iset = isets[pl]
            if iset not in pl_isets[pl]:
              pl_isets[pl][iset] = ids[pl]
              ids[pl] += 1
              iset_previous_action[pl].append(previous_info.actions[pl])
              iset_action_mask[pl].append(actions_mask)
              iset_action_depth[pl].append(previous_info.prev_actions[pl])
            depth_history_iset[pl][depth].append(pl_isets[pl][iset])
            depth_history_actions[pl][depth].append([
                i + pl_isets[pl][iset] * distinct_actions
                for i in range(distinct_actions)
            ])
          #Give invalid iset to the player that is not acting
          else:
            depth_history_iset[pl][depth].append(0)
            depth_history_actions[pl][depth].append(
                [0 for _ in range(distinct_actions)]
            )
      else:
        for pl in range(players):
          depth_history_iset[pl][depth].append(0)
          depth_history_actions[pl][depth].append(
              [0 for _ in range(distinct_actions)]
          )
      if state.terminal:
        return
      elif is_chance_node:
        new_info = PreviousInfo(
          previous_info.actions,
          previous_info.isets,
          previous_info.prev_actions,
          history_id,
          turns = previous_info.turns
        )
        pc_chance_outcomes = self.game.generate_all_public_card_nodes(state)
        for i, outcome in enumerate(pc_chance_outcomes):
        #4 chance outcomes
          next_history_temp[i] = (
                len(depth_history_player[depth + 1])
                if len(depth_history_player) > depth + 1
                else 0
            )
          _traverse_tree(outcome, new_info, initial_legals, 0.0, depth +1, chance = chance * 0.25)
        return
      for ai, a in enumerate(legal_actions[acting_player]):
        #print(state)
        if a < 0.5:
          continue
        #new_chance = chance * chance_probabilities[a]
        #assert new_chance > 0.0
        joint_action = [ai, 0] if acting_player == 0 else [0, ai]
        #print(joint_action)
        new_actions = tuple(
            pl_isets[pl][isets[pl]] * distinct_actions + ai if pl == acting_player else previous_info.actions[pl]
            for pl in range(players)
        )
        new_infosets = tuple(
            pl_isets[pl][isets[pl]] if pl == acting_player else previous_info.isets[pl]
            for pl in range(players)
        )
        new_prev_actions = tuple(
            previous_info.prev_actions[pl] + int(pl == acting_player)
            for pl in range(players)
        )
        new_info = PreviousInfo(
            new_actions,
            new_infosets,
            new_prev_actions,
            history_id,
            turns = previous_info.turns + 1
        )
        new_state, next_terminal, next_reward, new_legals = self.game.apply_action(
            state, self.dummy_key, previous_info.turns, jnp.array(joint_action))
        next_chance_node = state.public_card == 0 and new_state.public_card > 0
        next_history_temp[ai] = (
            len(depth_history_player[depth + 1])
            if len(depth_history_player) > depth + 1
            else 0
        )
        if not next_chance_node:
        # simple workaround if the next element was not visited yet
          _traverse_tree(new_state, new_info, new_legals, next_reward, depth + 1, chance = chance)
        else:
            # 4 chance outcomes in inner chance node
            chance_legals = [[1, 1, 1, 1], [1, 0, 0, 0]]
            _traverse_tree(new_state, new_info, chance_legals, next_reward, depth + 1, True, chance)
    #Just for checking subgames
    if start_state_info is not None:
      root_state, legals, turn = start_state_info
      _traverse_tree(
          root_state,
          PreviousInfo(
              tuple(0 for _ in range(players)),
              tuple(0 for _ in range(players)),
              tuple(0 for _ in range(players)),
              0,
              turn,
          ),
          legals,
          0.0,
          0,
          chance= 1
      )
    else:
      roots, legals = self.game.generate_all_private_card_nodes()
      for root_state in roots:
        _traverse_tree(
            root_state,
            PreviousInfo(
                tuple(0 for _ in range(players)),
                tuple(0 for _ in range(players)),
                tuple(0 for _ in range(players)),
                0,
                0,
            ),
            legals,
            0.0,
            0,
            chance= 1/30
        )

    def convert_to_jax(x):
      return [jnp.asarray(i) for i in x]

    def convert_to_jax_players(x):
      return [[jnp.asarray(i) for i in x[pl]] for pl in range(players)]
    

    depth_history_utility = convert_to_jax(depth_history_utility)
    depth_history_iset = convert_to_jax_players(depth_history_iset)
    depth_history_previous_iset = convert_to_jax_players(
        depth_history_previous_iset
    )
    depth_history_actions = convert_to_jax_players(depth_history_actions)
    depth_history_previous_action = convert_to_jax_players(
        depth_history_previous_action
    )
    depth_history_action_mask = convert_to_jax(depth_history_action_mask)

    depth_history_next_history = convert_to_jax(depth_history_next_history)
    depth_history_player = convert_to_jax(depth_history_player)
    depth_history_chance = convert_to_jax(depth_history_chance)
    depth_history_previous_history = convert_to_jax(
        depth_history_previous_history
    )
    depth_history_chance_probabilities = convert_to_jax(
        depth_history_chance_probabilities
    )

    #breakpoint()
    max_iset_depth = [np.max(iset_action_depth[pl]) for pl in range(players)]
    iset_previous_action = convert_to_jax(iset_previous_action)
    iset_action_mask = convert_to_jax(iset_action_mask)
    iset_action_depth = convert_to_jax(iset_action_depth)

    self.constants = SimultaneousJaxCFRConstants(
        players=players,
        max_depth=int(len(depth_history_utility)),
        max_actions=distinct_actions,
        max_iset_depth=max_iset_depth,
        isets=ids,
        depth_history_utility=depth_history_utility,
        depth_history_iset=depth_history_iset,
        depth_history_actions=depth_history_actions,
        depth_history_previous_iset=depth_history_previous_iset,
        depth_history_previous_action=depth_history_previous_action,
        depth_history_next_history=depth_history_next_history,
        depth_history_player=depth_history_player,
        depth_history_chance=depth_history_chance,
        depth_history_previous_history=depth_history_previous_history,
        depth_history_action_mask=depth_history_action_mask,
        depth_history_chance_probabilities=depth_history_chance_probabilities,
        iset_previous_action=iset_previous_action,
        iset_action_mask=iset_action_mask,
        iset_action_depth=iset_action_depth,
    )

    self.regrets = [
        jnp.zeros((ids[pl], distinct_actions)) for pl in range(players)
    ]
    self.averages = [
        jnp.zeros((ids[pl], distinct_actions)) for pl in range(players)
    ]

    self.regret_matching = jax.vmap(regret_matching, 0, 0)
    if self._regret_matching_plus:
      self.update_regrets = jax.vmap(update_regrets_plus, 0, 0)
    else:
      self.update_regrets = jax.vmap(update_regrets, 0, 0)

    self.iset_map = pl_isets

  def multiple_steps(self, iterations: int):
    """Performs several CFR steps.

    Args:
      iterations: Amount of CFR steps, the solver should do.
    """
    for _ in range(iterations):
      self.step()

  def evaluate_and_update_policy(self):
    """Wrapper to step().

    Ensures interchangability with
    open_spiel.python.algorithms.cfr._CFRSolverBase.
    """
    self.step()

  def step(self):
    """Wrapper around the jitted function for performing CFR step."""
    averaging_coefficient = self.timestep if self._linear_averaging else 1
    if self._alternating_updates:
      for player in range(self.constants.players):
        self.regrets, self.averages = self.jit_step(
            self.regrets, self.averages, averaging_coefficient, player
        )

    else:
      self.regrets, self.averages = self.jit_step(
          self.regrets,
          self.averages,
          averaging_coefficient,
          JAX_CFR_SIMULTANEOUS_UPDATE,
      )

    self.timestep += 1

  def propagate_strategy(self, current_strategies):
      """Propagtes the strategies withing infosets.

      Args:
        current_strategies: Current strategies for all players, list[Float[Isets,
          Actions]]
      Returns:
        realization_plans: the realization plans.
      """
      realization_plans = [
          jnp.ones_like(current_strategies[pl])
          for pl in range(self.constants.players)
      ]

      for pl in range(self.constants.players):
        for i in range(0, self.constants.max_iset_depth[pl] + 1):
          realization_plans[pl] = jnp.where(
              self.constants.iset_action_depth[pl][..., jnp.newaxis] == i,
              current_strategies[pl]
              * realization_plans[pl].ravel()[
                  self.constants.iset_previous_action[pl]
              ][..., jnp.newaxis],
              realization_plans[pl],
          )

      return realization_plans

  @functools.partial(jax.jit, static_argnums=(0,))
  def jit_step(
      self, regrets, averages, average_policy_update_coefficient, player
  ):
    """Performs the CFR step.

    This consists of:
    1. Computes the current strategies based on regrets
    2. Computes the realization plan for each action from top of the tree down
    3. Compute the counterfactual regrets from bottom of the tree up
    4. Updates regrets and average stretegies

    Args:
      regrets: Cummulative regrets for all players, list[Float[Isets, Actions]]
      averages: Average strategies for all players, list[Float[Isets, Actions]]
      average_policy_update_coefficient: Weight of the average policy update.
        When enabled linear_averging it is equal to current iteration. Otherwise
        1, int
      player: Player for which the update should be done. When alternating
        updates are distables, it is JAX_CFR_SIMULTANEOUS_UPDATE

    Returns:
      regrets: the regrets.
      averages: the averages.
    """
    current_strategies = [
        self.regret_matching(regrets[pl], self.constants.iset_action_mask[pl])
        for pl in range(self.constants.players)
    ]

    realization_plans = self.propagate_strategy(current_strategies)
    iset_reaches = [
        jnp.sum(realization_plans[pl], -1)
        for pl in range(self.constants.players)
    ]
    # In last row, there are only terminal, so we start row before it
    depth_utils = [
        [self.constants.depth_history_utility[-1] * (1 - 2 * pl)]
        for pl in range(self.constants.players)
    ]
    for i in range(self.constants.max_depth - 2, -1, -1):

      each_history_policy = self.constants.depth_history_chance_probabilities[i]
      for pl in range(self.constants.players):
        each_history_policy = each_history_policy * jnp.where(
            self.constants.depth_history_player[i][..., jnp.newaxis] == pl,
            current_strategies[pl][self.constants.depth_history_iset[pl][i]],
            1,
        )

      for pl in range(self.constants.players):
        action_value = jnp.where(
            self.constants.depth_history_player[i][..., jnp.newaxis] == -4,
            self.constants.depth_history_utility[i][..., jnp.newaxis] * (1 - 2 * pl),
            depth_utils[pl][-1][self.constants.depth_history_next_history[i]],
        )
        history_value = jnp.sum(action_value * each_history_policy, -1)
        regret = (
            (action_value - history_value[..., jnp.newaxis])
            * self.constants.depth_history_action_mask[i]
            * (self.constants.depth_history_player[i][..., jnp.newaxis] == pl)
            * self.constants.depth_history_chance[i][..., jnp.newaxis]
        )
        for pl2 in range(self.constants.players):
          if pl != pl2:
            regret = (
                regret
                * realization_plans[pl2].ravel()[
                    self.constants.depth_history_previous_action[pl2][i]
                ][..., jnp.newaxis]
            )
        bin_regrets = jnp.bincount(
            self.constants.depth_history_actions[pl][i].ravel(),
            regret.ravel(),
            length=self.constants.isets[pl] * self.constants.max_actions,
        )
        bin_regrets = bin_regrets.reshape(-1, self.constants.max_actions)
        regrets[pl] = jnp.where(
            jnp.logical_or(player == pl, player == JAX_CFR_SIMULTANEOUS_UPDATE),
            regrets[pl] + bin_regrets,
            regrets[pl],
        )
        #depth_utils[pl].append(history_value)
        depth_utils[pl][-1] = history_value

    regrets = [
        self.update_regrets(regrets[pl]) for pl in range(self.constants.players)
    ]

    averages = [
        jnp.where(
            jnp.logical_or(player == pl, player == JAX_CFR_SIMULTANEOUS_UPDATE),
            averages[pl]
            + current_strategies[pl]
            * iset_reaches[pl][..., jnp.newaxis]
            * average_policy_update_coefficient,
            averages[pl],
        )
        for pl in range(self.constants.players)
    ]

    return regrets, averages

  def average_policy(self):
    """Extracts the average policy from JAX structures into a JaxPolicy."""
    averages = [
        np.asarray(self.averages[pl]) for pl in range(self.constants.players)
    ]
    averages = [
      np.where(averages[pl] >= 0, averages[pl], np.ones_like(averages[pl])) for pl in range(self.constants.players)
    ]
    averages = [
        averages[pl] / np.sum(averages[pl], -1, keepdims=True)
        for pl in range(self.constants.players)
    ]
    #breakpoint()
    avg_strategy = JaxPolicy()

    for pl in range(2):
      for iset, idx in self.iset_map[pl].items():
        if not iset:
          continue
        state_policy = np.zeros_like(averages[pl][idx])
        for i in range(len(state_policy)):
          state_policy[i] = averages[pl][idx][i]
        avg_strategy[iset] = state_policy
    return avg_strategy

