"""
Supervised Learning on generated training data.
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

	# You can continue training by setting "continue_training" to True
	args = {"type": "normal", "cmd": "opt", "continue_training": False}
	manager.start(args)
