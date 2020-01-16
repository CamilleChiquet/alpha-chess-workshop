"""
Generates training data from .pgn files in data/play_data/ for the supervised learning phase.
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

	args = {"type": "normal", "cmd": "sl"}
	manager.start(args)
