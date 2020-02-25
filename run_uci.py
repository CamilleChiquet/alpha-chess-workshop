"""
Script used by the AlphaZero.bat when called by a software like Arena.
Read the readme.md to learn how to play against the AI with a GUI (Graphical User Interface).
"""
import multiprocessing as mp
import os
import sys
import keras

_PATH_ = os.path.dirname(os.path.dirname(__file__))

if _PATH_ not in sys.path:
	sys.path.append(_PATH_)

if __name__ == "__main__":
	mp.set_start_method('spawn')
	sys.setrecursionlimit(10000)
	import manager

	keras.backend.set_learning_phase(0)
	manager.start(worker="uci", config_type="normal")