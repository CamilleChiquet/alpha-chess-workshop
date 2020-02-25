"""
Contains the set of configs to use for the "normal" version of the app.
"""


class EvaluateConfig:
	def __init__(self):
		self.game_num = 100
		self.replace_rate = 0.55
		self.play_config = PlayConfig()
		self.play_config.simulation_num_per_move = 200
		self.play_config.c_puct = 1  # lower  = prefer mean action value
		self.play_config.tau_decay_rate = 0.6  # I need a better distribution...
		self.play_config.noise_eps = 0
		self.evaluate_latest_first = True
		self.max_game_length = 200


class PlayDataConfig:
	def __init__(self):
		self.min_elo_policy = 500  # 0 weight
		self.max_elo_policy = 1_800  # 1 weight
		self.nb_game_in_file = 250
		self.nb_game_between_training_sessions = 1_000


class PlayConfig:
	def __init__(self):
		self.max_processes = 1
		self.search_threads = 16
		self.simulation_num_per_move = 200
		self.c_puct = 1.5
		self.noise_eps = 0.25
		self.dirichlet_alpha = 0.3
		self.tau_decay_rate = 1.0
		self.zero_temperature_half_move = 30
		self.virtual_loss = 3
		self.resign_threshold = -0.8
		self.min_resign_turn = 5
		self.max_game_length = 200


class TrainerConfig:
	def __init__(self):
		self.cleaning_processes = 6  # RAM explosion...
		self.nb_recent_files_for_training = 800  # For reinforcement learning pipeline

		self.batch_size = 400  # tune this to your gpu memory

		self.dataset_size = 100_000

		# Learning rate parameters
		self.start_lr = 1e-4
		self.min_lr = 1e-4
		# If there is no improvement in the last X iteration(s), then the learning rate is lowered
		self.loss_patience = 25

		# When training has started, enter the following command in a terminal
		# > tensorboard --logdir=[replace by your logs directory]
		self.tensorboard_enabled = True

		# Set it to True if you want to continue a previously stopped training.
		# It will take the latest saved model in the "data/model/next_generation" directory and continue the
		# training from this point.
		self.continue_training = False


class ModelConfig:
	cnn_filter_num = 256
	cnn_first_filter_size = 5
	cnn_filter_size = 3
	common_res_layer_num = 20
	policy_res_layer_num = 10
	value_res_layer_num = 10
	value_fc_size = 256
	input_depth = 18
