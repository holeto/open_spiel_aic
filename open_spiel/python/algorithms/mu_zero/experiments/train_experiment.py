
from open_spiel.python.algorithms.mu_zero.mu_zero_train import MuZeroTrain, MuZeroTrainConfig
from open_spiel.python.algorithms.mu_zero.experiments.utils import save_model

import os

def train(args, game, trajectory_max, save_folder):
  config = MuZeroTrainConfig(
    batch_size=args.batch_size,
    trajectory_max=trajectory_max,
    sampling_epsilon=args.sampling_epsilon,
    
    train_rnad=args.train_rnad,
    train_mvs=args.train_mvs,
    train_abstraction=args.train_abstraction,
    train_dynamics=args.train_dynamics,
    train_legal_actions=args.train_legal_actions,
    
    use_abstraction=args.use_abstraction,
    abstraction_amount=args.abstraction_amount,
    abstraction_size=args.abstraction_size,
    similarity_metric=args.similarity_metric,
    
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
  
  for iteration in range(args.iterations):
    train_algorithm.multiple_goofspiel_steps(args.save_each)
    file_name = save_folder + "muzero_" + str(iteration) + ".pkl" 
    print("Saving iteration", iteration, flush=True)
    save_model(file_name, train_algorithm)
    