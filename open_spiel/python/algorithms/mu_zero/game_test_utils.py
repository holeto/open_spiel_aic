
from collections import deque
#For each iset and public state get how many
#states fall under it
def extract_from_spiel(game, provides_public_state=False):
  #state_tensors = []
  count_states_by_isets = [{} for _ in range(2)]
  count_states_by_public_state = {}
  q = deque()
  visited = []
  init_state = game.new_initial_state()
  q.append(init_state)
  while len(q) > 0:
    state = q.popleft()
    if state.is_terminal():
      continue
    if state.is_chance_node():
      outcomes = [action for action, probability in state.chance_outcomes()]
      for a in outcomes:
        new_state = state.clone()
        new_state.apply_action(a)
        #if not new_state.is_terminal():
        q.append(new_state)
      continue
    public_state = str(state.public_state_tensor()) if provides_public_state else "0"
    if(public_state in count_states_by_public_state.keys()):
      count_states_by_public_state[public_state] += 1
    else:
      count_states_by_public_state[public_state] = 1
    for pl in range(2):
      pl_iset = state.information_state_string(pl)
      if(pl_iset in count_states_by_isets[pl].keys()):
        #print("Infoset match for player ", pl)
        #print(str(state))
        #print(pl_iset)
        count_states_by_isets[pl][pl_iset] += 1
      else:
        count_states_by_isets[pl][pl_iset] = 1
    if not str(state) in visited:
      #print("At state ", str(state))
      #print(len(state_tensors))
      #print(len(state.state_tensor()))
      visited.append(str(state)) 
      for a in state.legal_actions():
        new_state = state.clone()
        new_state.apply_action(a)
        #if not new_state.is_terminal():
        q.append(new_state)
  return count_states_by_isets[0].values(), count_states_by_isets[1].values(), count_states_by_public_state.values()

def histogram(values):
  hist = []
  for value in values:
    #we start the histogram at 1 here
    if value > len(hist):
      for _ in range(value - len(hist)):
        hist.append(0)
    hist[value - 1] += 1
  return hist

def compare_hists(hist1, hist2):
  for i, vals in enumerate(zip(hist1, hist2)):
    val1, val2 = vals
    if val1 != val2:
      print("Hists differ at index", i)
      return
  print("Hists match")