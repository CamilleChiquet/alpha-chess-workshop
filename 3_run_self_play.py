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

	# The "data/model/model_best_weight.h5" model will be loaded. Make sure it is the one you want to use.
	# If it doesn't exist, a model with random weights and biases will be used to play.
	args = {"type": "normal", "cmd": "self"}
	manager.start(args)
