"""
Make to networks compete against each other.
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

	white_model_path = "data/model/last"
	black_model_path = "data/model/old"
	# model_1 is white, model_2 plays as black player
	manager.start(worker="duel", config_type="normal", model_1_path=white_model_path, model_2_path=black_model_path,
	              deterministic=False)