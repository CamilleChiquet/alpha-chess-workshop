"""
Various helper functions for working with the data used in this app
"""

import os
import pickle
from datetime import datetime
from glob import glob
from logging import getLogger

import chess

from config import ResourceConfig

logger = getLogger(__name__)


def pretty_print(env, colors):
	new_pgn = open("game.pgn", "at")
	game = chess.pgn.Game.from_board(env.board)
	game.headers["Result"] = env.result
	game.headers["White"], game.headers["Black"] = colors
	game.headers["Date"] = datetime.now().strftime("%Y.%m.%d")
	new_pgn.write(str(game) + "\n\n")
	new_pgn.close()


def find_pgn_files(directory, pattern='*.pgn'):
	dir_pattern = os.path.join(directory, pattern)
	files = list(sorted(glob(dir_pattern)))
	return files


def get_game_data_filenames(rc: ResourceConfig):
	file_ext = "_fen.pickle"
	pattern = os.path.join(rc.play_data_dir, rc.play_data_filename_tmpl % "*" + file_ext)
	files = list(sorted(glob(pattern)))
	for i in range(len(files)):
		files[i] = files[i][:-len(file_ext)]
	return files


def get_next_generation_model_dirs(rc: ResourceConfig):
	dir_pattern = os.path.join(rc.next_generation_model_dir, rc.next_generation_model_dirname_tmpl % "*")
	dirs = list(sorted(glob(dir_pattern)))
	return dirs


def save_as_pickle_object(path, data):
	with open(path, 'wb') as f:
		pickle.dump(data, f)


def read_pickle_object(path):
	with open(path, "rb") as f:
		return pickle.load(f)
