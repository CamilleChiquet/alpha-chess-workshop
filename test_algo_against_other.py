"""
Make the network play against itself and records the games inside a game.pgn file.

The "data/model/model_best_weight.h5" model will be loaded. Make sure it is the one you want to use.
If it doesn't exist, a model with random weights and biases will be used to play.
"""
import multiprocessing as mp
import os
import sys

_PATH_ = os.path.dirname(os.path.dirname(__file__))

if _PATH_ not in sys.path:
	sys.path.append(_PATH_)

if __name__ == "__main__":
	mp.set_start_method('spawn')
	sys.setrecursionlimit(10000)
	import manager

	# model_1 is white, model_2 plays as black player
	manager.start(worker="duel", config_type="normal", model_1_path="data/model/new", model_2_path="data/model/old",
	              deterministic=True)