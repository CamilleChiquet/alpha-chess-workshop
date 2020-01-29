"""
Holds the worker which trains the chess model using self play data.
"""
from logging import getLogger
from multiprocessing import Manager
from time import time

import numpy as np

from agent.model_chess import ChessModel
from agent.player_chess import ChessPlayer
from config import Config, PlayWithHumanConfig
from env.chess_env import ChessEnv, Winner
from lib.data_helper import pretty_print

logger = getLogger(__name__)


def start(config: Config, model_1_path: str, model_2_path: str, deterministic: bool = False):
	return DuelWorker(config, model_1_path, model_2_path, deterministic=deterministic).start()


class DuelWorker:
	"""
	Worker which trains a chess model using self play data. ALl it does is do self play and then write the
	game data to file, to be trained on by the optimize worker.

	Attributes:
		:ivar Config config: config to use to configure this worker
		:ivar ChessModel current_model_1: model that will play as white player
		:ivar ChessModel current_model_2: model that will play as black player
		:ivar Manager m: the manager to use to coordinate between other workers
		:ivar list(Connection) cur_pipes(1/2): pipes to send observations to and get back mode predictions.
		:ivar list((str,list(float))): list of all the moves. Each tuple has the observation in FEN format and
			then the list of prior probabilities for each action, given by the visit count of each of the states
			reached by the action (actions indexed according to how they are ordered in the uci move list).
	"""

	def __init__(self, config: Config, model_1_path: str, model_2_path: str, deterministic: bool = False):
		self.config = config
		self.config.play.simulation_num_per_move = 800
		self.config.play.noise_eps = 0

		if deterministic:
			self.config.play.c_puct = 1 # lower  = prefer mean action value
			self.config.play.tau_decay_rate = 0  # start deterministic mode
			self.config.play.resign_threshold = None

		self.current_model_1 = self.load_model(model_1_path)
		self.current_model_2 = self.load_model(model_2_path)

		self.m = Manager()
		self.cur_pipes_1 = self.m.list([self.current_model_1.get_pipes(self.config.play.search_threads) for _ in
		                                range(self.config.play.max_processes)])
		self.cur_pipes_2 = self.m.list([self.current_model_2.get_pipes(self.config.play.search_threads) for _ in
		                                range(self.config.play.max_processes)])

	def start(self):
		"""
		Do self play and write the data to the appropriate file.
		"""

		game_idx = 0
		while True:
			game_idx += 1
			start_time = time()
			env, fen_data, moves_array, scores_array = play_buffer(config=self.config,
			                                                       cur=(self.cur_pipes_1, self.cur_pipes_2))
			print(f"game {game_idx:3} time={time() - start_time:5.1f}s halfmoves={env.num_halfmoves:3} {env.winner:12} "
			      f"{'by resign ' if env.resigned else ''}")

			pretty_print(env, ("current_model", "current_model"))

	def load_model(self, model_path: str):
		"""
		Load the current best model
		:return ChessModel: current best model
		"""
		model = ChessModel(self.config)
		model_config_path = model_path + ".json"
		model_weights_path = model_path + ".h5"
		if not model.load(model_config_path, model_weights_path):
			raise IOError(f"Failed to load {model_config_path} or {model_weights_path}")
		return model


def play_buffer(config, cur) -> (ChessEnv, list):
	"""
	Play one game and add the play data to the buffer
	:param Config config: config for how to play
	:param list(Connection) cur: list of pipes to use to get a pipe to send observations to for getting
		predictions. One will be removed from this list during the game, then added back
	:return (ChessEnv,list((str,list(float)): a tuple containing the final ChessEnv state and then a list
		of data to be appended to the DuelWorker.buffer
	"""
	pipes_1 = cur[0].pop()  # borrow
	pipes_2 = cur[0].pop()  # borrow
	env = ChessEnv().reset()

	white = ChessPlayer(config, pipes=pipes_1)
	black = ChessPlayer(config, pipes=pipes_2)

	while not env.done:
		if env.white_to_move:
			action = white.action(env)
		else:
			action = black.action(env)
		env.step(action)
		if env.num_halfmoves >= config.play.max_game_length:
			env.adjudicate()

	if env.winner == Winner.white:
		black_win = -1
	elif env.winner == Winner.black:
		black_win = 1
	else:
		black_win = 0

	black.finish_game(black_win)
	white.finish_game(-black_win)

	cur[0].append(pipes_1)
	cur[1].append(pipes_2)

	fen_data = []
	moves_array = np.zeros((len(white.moves) + len(black.moves), white.labels_n), dtype=np.float16)
	scores = np.zeros((len(white.moves) + len(black.moves)), dtype=np.int8)
	for i in range(len(white.moves)):
		fen_data.append(white.moves[i][0])
		moves_array[i * 2] = white.moves[i][1]
		scores[i * 2] = white.moves[i][2]
		if i < len(black.moves):
			fen_data.append(black.moves[i][0])
			moves_array[i * 2 + 1] = black.moves[i][1]
			scores[i * 2 + 1] = black.moves[i][2]

	return env, fen_data, moves_array, scores
