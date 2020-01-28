"""
Reinforcement Learning phase.

Put your best model in "data/model/" and name it model_best_weight.h5.
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

	manager.start(worker="self", config_type="normal")
