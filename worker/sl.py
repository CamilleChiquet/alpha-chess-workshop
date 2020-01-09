"""
Contains the worker for training the model using recorded game data rather than self-play
"""
import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from logging import getLogger
from time import time

import chess.pgn
import numpy as np

from agent.player_chess import ChessPlayer
from config import Config
from env.chess_env import ChessEnv, Winner
from lib.data_helper import save_as_pickle_object, find_pgn_files

logger = getLogger(__name__)

TAG_REGEX = re.compile(r"^\[([A-Za-z0-9_]+)\s+\"(.*)\"\]\s*$")


def start(config: Config):
	return SupervisedLearningWorker(config).start()


class SupervisedLearningWorker:
	"""
	Worker which performs supervised learning on recorded games.

	Attributes:
		:ivar Config config: config for this worker
		:ivar list((str,list(float)) buffer: buffer containing the data to use for training -
			each entry contains a FEN encoded game state and a list where every index corresponds
			to a chess move. The move that was taken in the actual game is given a value (based on
			the player elo), all other moves are given a 0.
	"""

	def __init__(self, config: Config):
		"""
		:param config:
		"""
		self.config = config
		self.fen_buffer = []
		self.moves_buffer = []
		self.scores_buffer = []

	def start(self):
		"""
		Start the actual training.
		"""
		self.fen_buffer = []
		self.moves_buffer = []
		self.scores_buffer = []
		self.idx = 0
		with ProcessPoolExecutor(max_workers=7) as executor:
			files = find_pgn_files(self.config.resource.play_data_dir)
			print(files)
			for filename in files:
				games = get_games_from_file(filename)
				print("done reading")
				for res in as_completed([executor.submit(get_buffer, self.config, game) for game in
				                         games]):  # poisoned reference (memleak)
					self.idx += 1
					env, fen_data, moves_array, scores_array = res.result()
					self.save_data(fen_data, moves_array, scores_array)

		if len(self.fen_buffer) > 0:
			self.flush_buffer()

	def save_data(self, fen_data: list, moves_array: np.ndarray, scores_array: np.ndarray):
		"""

		:param (str,list(float)) data: a FEN encoded game state and a numpy array where every index corresponds
			to a chess move. The move that was taken in the actual game is given a value (based on
			the player elo), all other moves are given a 0.
		"""
		self.fen_buffer.append(fen_data)
		self.moves_buffer.append(moves_array)
		self.scores_buffer.append(scores_array)
		if self.idx % self.config.play_data.sl_nb_game_in_file == 0:
			self.flush_buffer()

	def flush_buffer(self):
		"""
		Clears out the moves loaded into the buffer and saves the to file.
		"""
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


def get_games_from_file(filename):
	"""

	:param str filename: file containing the pgn game data
	:return list(pgn.Game): chess games in that file
	"""
	pgn = open(filename, errors='ignore')
	offsets = list(chess.pgn.scan_offsets(pgn))
	n = len(offsets)
	print(f"found {n} games")
	games = []
	for offset in offsets:
		pgn.seek(offset)
		games.append(chess.pgn.read_game(pgn))
	return games


def clip_elo_policy(config, elo):
	return min(1, max(0, elo - config.play_data.min_elo_policy) / config.play_data.max_elo_policy)


# 0 until min_elo, 1 after max_elo, linear in between


def get_buffer(config, game) -> (ChessEnv, list):
	"""
	Gets data to load into the buffer by playing a game using PGN data.
	:param Config config: config to use to play the game
	:param pgn.Game game: game to play
	:return list(str,list(float)): data from this game for the SupervisedLearningWorker.buffer
	"""
	env = ChessEnv().reset()
	white = ChessPlayer(config, dummy=True)
	black = ChessPlayer(config, dummy=True)
	result = game.headers["Result"]
	white_elo, black_elo = int(game.headers["WhiteElo"]), int(game.headers["BlackElo"])
	white_weight = clip_elo_policy(config, white_elo)
	black_weight = clip_elo_policy(config, black_elo)
	# TODO : try with weights of 1, whatever the elo rank is

	actions = []
	while not game.is_end():
		game = game.variation(0)
		actions.append(game.move.uci())
	k = 0
	while not env.done and k < len(actions):
		if env.white_to_move:
			action = white.sl_action(env.observation, actions[k], weight=white_weight)  # ignore=True
		else:
			action = black.sl_action(env.observation, actions[k], weight=black_weight)  # ignore=True
		env.step(action, False)
		k += 1

	if not env.board.is_game_over() and result != '1/2-1/2':
		env.resigned = True
	if result == '1-0':
		env.winner = Winner.white
		black_win = -1
	elif result == '0-1':
		env.winner = Winner.black
		black_win = 1
	else:
		env.winner = Winner.draw
		black_win = 0

	black.finish_game(black_win)
	white.finish_game(-black_win)

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
