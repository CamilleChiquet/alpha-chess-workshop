"""
Contains the set of configs to use for the "normal" version of the app.
"""


class EvaluateConfig:
	def __init__(self):
		self.game_num = 50
		self.replace_rate = 0.55
		self.play_config = PlayConfig()
		self.play_config.simulation_num_per_move = 800
		self.play_config.thinking_loop = 1
		self.play_config.c_puct = 1  # lower  = prefer mean action value
		self.play_config.tau_decay_rate = 0.6  # I need a better distribution...
		self.play_config.noise_eps = 0
		self.evaluate_latest_first = True
		self.max_game_length = 1_000


class PlayDataConfig:
	def __init__(self):
		self.min_elo_policy = 500  # 0 weight
		self.max_elo_policy = 1_800  # 1 weight
		self.sl_nb_game_in_file = 250
		self.nb_game_in_file = 50


class PlayConfig:
	def __init__(self):
		self.max_processes = 3
		self.search_threads = 16
		self.simulation_num_per_move = 1600
		self.thinking_loop = 1
		self.logging_thinking = False
		self.c_puct = 1.5
		self.noise_eps = 0.25
		self.dirichlet_alpha = 0.3
		self.tau_decay_rate = 0.99
		self.virtual_loss = 3
		self.resign_threshold = -0.8
		self.min_resign_turn = 5
		self.max_game_length = 1_000


class TrainerConfig:
	def __init__(self):
		self.cleaning_processes = 5  # RAM explosion...
		self.batch_size = 512  # tune this to your gpu memory
		self.epoch_to_checkpoint = 1
		self.dataset_size = 100_000


class ModelConfig:
	cnn_filter_num = 256
	cnn_first_filter_size = 5
	cnn_filter_size = 3
	res_layer_num = 20
	value_fc_size = 256
	input_depth = 18
