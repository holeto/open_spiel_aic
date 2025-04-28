from open_spiel.python.algorithms.mu_zero.jax_games.jax_game_algorithms import nash_equilibrium_jax_game
from open_spiel.python.algorithms.mu_zero.mu_zero_train import MuZeroTrain, MuZeroTrainConfig
from open_spiel.python.algorithms.mu_zero.experiments.utils import save_model, stringify, destringify

from open_spiel.python.algorithms.mu_zero.muzero_networks import RNaDNetwork
from open_spiel.python.algorithms.mu_zero.flax_utils import optax_optimizer


import os
import numpy as np
import jax
import jax.numpy as jnp
import optax

def train(args, game, trajectory_max, save_folder):
  
    
  config = MuZeroTrainConfig(
    batch_size=args.batch_size,
    trajectory_max=trajectory_max,
    sampling_epsilon=args.sampling_epsilon,
    
    train_rnad=args.train_rnad,
    train_mvs=args.train_mvs,
    train_transformations=args.train_transformations,
    train_abstraction=args.train_abstraction,
    train_dynamics=args.train_dynamics,
    train_legal_actions=args.train_legal_actions,
    
    use_abstraction=args.use_abstraction,
    abstraction_amount=args.abstraction_amount,
    abstraction_size=args.abstraction_size,
    similarity_metric=args.similarity_metric,
    
    
    abstraction_soft_k_means_temperature=args.abstraction_soft_k_means_temperature,
    abstraction_soft_k_means_closeness_assignment=args.abstraction_soft_k_means_closeness_assignment,
    abstraction_soft_k_means_repulsive_force=args.abstraction_soft_k_means_repulsive_force,
    transformation_soft_k_means_temperature=args.transformation_soft_k_means_temperature,
    transformation_soft_k_means_closeness_assignment=args.transformation_soft_k_means_closeness_assignment,
    transformation_soft_k_means_repulsive_force=args.transformation_soft_k_means_repulsive_force,
    
    dynamics_type=args.dynamics_type,
    
    ps_encoder_hidden_size=args.ps_encoder_hidden_size,
    ps_decoder_hidden_size=args.ps_decoder_hidden_size,
    iset_hidden_size=args.iset_hidden_size,
    dynamics_hidden_size=args.dynamics_hidden_size,
    similarity_hidden_size=args.similarity_hidden_size,
    mvs_hidden_size=args.mvs_hidden_size,
    legal_actions_hidden_size=args.legal_actions_hidden_size,
    rnad_hidden_size=args.rnad_hidden_size,
    
    transformations=args.transformations,
    matrix_valued_states=args.matrix_valued_states,
    
    c_iset_vtrace=args.c_iset_vtrace,
    rho_iset_vtrace=args.rho_iset_vtrace,
    c_state_vtrace=args.c_state_vtrace,
    rho_state_vtrace=args.rho_state_vtrace,
    
    eta_regularization=args.eta_regularization,
    entropy_schedule_repeats=args.entropy_schedule_repeats,
    entropy_schedule_size=args.entropy_schedule_size,
    
    learning_rate=args.learning_rate,
    target_network_update=args.target_network_update,
    seed=args.seed
  )
  train_algorithm = MuZeroTrain(game, config)
   
  if not os.path.exists(save_folder):
    os.makedirs(save_folder)
  
  for iteration in range(args.iterations + 1):
    file_name = save_folder + "muzero_" + str(iteration) + ".pkl" 
    print("Saving iteration", iteration, flush=True)
    save_model(file_name, train_algorithm)
    train_algorithm.multiple_jax_steps(args.save_each)
    
    
def continue_training(train_algorithm: MuZeroTrain, save_folder: str, save_each:int,  first_iteration: int, iterations: int):
  
  if not os.path.exists(save_folder):
    os.makedirs(save_folder)
  
  # We add +1 to first iteration to ensure the old model is not rewritten.
  for iteration in range(first_iteration + 1, first_iteration + iterations + 2):
    file_name = save_folder + "muzero_" + str(iteration) + ".pkl" 
    print("Saving iteration", iteration, flush=True)
    save_model(file_name, train_algorithm)
    train_algorithm.multiple_jax_steps(save_each)
    
  
  
  

def train_nash(args, game, save_folder):
  
  _, nash_policy, nash_value = nash_equilibrium_jax_game(game, 2000)
  
  iset_input = []
  legal_input = []
  net_output = []
  for iset_str, policy in nash_policy.policy.items():
    iset = destringify(iset_str)
    policy_np = np.array(policy)
    iset_input.append(iset)
    net_output.append(policy_np)
    
    legals = (policy_np > 1e-9).astype(np.int32)
    legal_input.append(legals)
   
  iset_input = np.array(iset_input)
  legal_input = np.array(legal_input)   
  net_output = np.array(net_output)
   
  network = RNaDNetwork(256, game.num_distinct_actions()) 
  init_key = jax.random.key(484)
  params = network.init(init_key, iset_input[0], legal_input[0])
  
  optimizer = optax_optimizer(params, optax.chain(optax.adam(3e-4), optax.clip(100)))
  
  def loss_function(params, isets, legals, policy):
    pi, _, _, _ = network.apply(params, isets, legals)
    loss = jnp.mean((pi - policy) ** 2)
    return loss
  
  loss_fn = jax.value_and_grad(loss_function, has_aux=False)
  
  @jax.jit
  def move_weights(params, optimizer, isets, legals, policy):
    loss, grads = loss_fn(params, isets, legals, policy)
    params = optimizer(params, grads)
    return params, optimizer, loss
  
  for i in range(20000):
    params, optimizer, loss = move_weights(params, optimizer, iset_input, legal_input, net_output)
    # print(loss)
    
  init_state, init_legals = game.initialize_structures(init_key)
  
  _, init_p1_iset, init_p2_iset, init_public_state = game.get_info(init_state)
  init_p1_iset = jnp.array(init_p1_iset)
  init_p2_iset = jnp.array(init_p2_iset) 
  
  
  
  print(network.apply(params, init_p1_iset, init_legals[0])[0])
  print(nash_policy.policy[stringify(np.array(init_p1_iset))])

  print(network.apply(params, init_p2_iset, init_legals[1])[0])
  print(nash_policy.policy[stringify(np.array(init_p2_iset))])
  
  
  

# from open_spiel.python.algorithms.mu_zero.jax_games.jax_goofspiel import JaxGoofspiel
# train_nash(None, JaxGoofspiel(5), "")