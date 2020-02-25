"""
Holds the worker which trains the chess model using self play data.
"""
import os
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from logging import getLogger
from multiprocessing import Manager
from time import time

import numpy as np
import keras

from agent.model_chess import ChessModel
from agent.player_chess import ChessPlayer
from config import Config
from env.chess_env import ChessEnv, Winner
from lib.data_helper import save_as_pickle_object, pretty_print
from lib.model_helper import load_best_model_weight, save_as_best_model

logger = getLogger(__name__)


def start(config: Config):
	return SelfPlayWorker(config).start()


class SelfPlayWorker:
	"""
	Worker which trains a chess model using self play data. ALl it does is do self play and then write the
	game data to file, to be trained on by the optimize worker.
	Attributes:
		:ivar Config config: config to use to configure this worker
		:ivar ChessModel current_model: model to use for self play
		:ivar Manager m: the manager to use to coordinate between other workers
		:ivar list(Connection) cur_pipes: pipes to send observations to and get back mode predictions.
		:ivar list((str,list(float))): list of all the moves. Each tuple has the observation in FEN format and
			then the list of prior probabilities for each action, given by the visit count of each of the states
			reached by the action (actions indexed according to how they are ordered in the uci move list).
	"""

	def __init__(self, config: Config):
		self.config = config
		self.current_model = self.load_model()
		self.m = Manager()
		self.cur_pipes = self.m.list([self.current_model.get_pipes(self.config.play.search_threads) for _ in
		                              range(self.config.play.max_processes)])
		self.fen_buffer = []
		self.moves_buffer = []
		self.scores_buffer = []

	def start(self):
		"""
		Do self play and write the data to the appropriate file.
		"""
		keras.backend.set_learning_phase(0)

		self.fen_buffer = []
		self.moves_buffer = []
		self.scores_buffer = []

		futures = deque()
		with ProcessPoolExecutor(max_workers=self.config.play.max_processes) as executor:
			for game_idx in range(self.config.play.max_processes * 2):
				futures.append(executor.submit(self_play_buffer, self.config, cur=self.cur_pipes))
			game_idx = 0
			while True:
				game_idx += 1
				start_time = time()
				env, fen_data, moves_array, scores_array = futures.popleft().result()
				print(f"game {game_idx:3} time={time() - start_time:5.1f}s "
				      f"halfmoves={env.num_halfmoves:3} {env.winner:12} "
				      f"{'by resign ' if env.resigned else '          '}")

				pretty_print(env, ("current_model", "current_model"))
				self.fen_buffer.append(fen_data)
				self.moves_buffer.append(moves_array)
				self.scores_buffer.append(scores_array)
				if (game_idx % self.config.play_data.nb_game_in_file) == 0:
					self.flush_buffer()
				if game_idx >= self.config.play_data.nb_game_between_training_sessions:
					break
				futures.append(executor.submit(self_play_buffer, self.config, cur=self.cur_pipes))  # Keep it going

		self.flush_buffer()

		keras.backend.set_learning_phase(1)

	def load_model(self):
		"""
		Load the current best model
		:return ChessModel: current best model
		"""
		model = ChessModel(self.config)
		if not load_best_model_weight(model):
			model.build()
			save_as_best_model(model)
		return model

	def flush_buffer(self):
		"""
		Flush the play data buffer and write the data to the appropriate location
		"""
		if len(self.fen_buffer) == 0:
			return

		rc = self.config.resource
		game_id = datetime.now().strftime("%Y%m%d-%H%M%S.%f")
		path = os.path.join(rc.play_data_dir, rc.play_data_filename_tmpl % game_id)
		logger.info(f"save play data to {path}")
		save_as_pickle_object(path + "_fen.pickle", self.fen_buffer)
		np.save(path + "_moves.npy", self.moves_buffer)
		np.save(path + "_scores.npy", self.scores_buffer)
		del self.fen_buffer, self.moves_buffer, self.scores_buffer
		self.fen_buffer = []
		self.moves_buffer = []
		self.scores_buffer = []


def self_play_buffer(config, cur) -> (ChessEnv, list):
	"""
	Play one game and add the play data to the buffer
	:param Config config: config for how to play
	:param list(Connection) cur: list of pipes to use to get a pipe to send observations to for getting
		predictions. One will be removed from this list during the game, then added back
	:return (ChessEnv,list((str,list(float)): a tuple containing the final ChessEnv state and then a list
		of data to be appended to the SelfPlayWorker.buffer
	"""
	pipes = cur.pop()  # borrow
	env = ChessEnv().reset()

	white = ChessPlayer(config, pipes=pipes)
	black = ChessPlayer(config, pipes=pipes)

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

	cur.append(pipes)

	fen_data = []
	moves_array = np.zeros((len(white.moves) + len(black.moves), white.labels_n), dtype=np.float16)
	scores = np.zeros((len(white.moves) + len(black.moves)), dtype=np.int8)
	for i in range(len(white.moves)):
		fen_data.append(white.moves[i][0])
		moves_array[i*2] = white.moves[i][1]
		scores[i*2] = white.moves[i][2]
		if i < len(black.moves):
			fen_data.append(black.moves[i][0])
			moves_array[i*2 + 1] = black.moves[i][1]
			scores[i*2 + 1] = black.moves[i][2]

	return env, fen_data, moves_array, scores