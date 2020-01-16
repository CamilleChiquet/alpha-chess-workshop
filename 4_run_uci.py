"""
Script used by the C0uci.bat when called by a software like Arena.
Read the readme.md to learn how to play against the AI with a GUI (Graphical User Interface).
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

	args = {"type": "normal", "cmd": "uci"}
	manager.start(args)