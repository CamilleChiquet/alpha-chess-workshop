"""
Manages starting off each of the separate processes involved in ChessZero -
self play, training, and evaluation.
"""
import argparse
from logging import getLogger, disable

from config import Config
from lib.logger import setup_logger

logger = getLogger(__name__)

CMD_LIST = ['self', 'opt', 'eval', 'sl', 'uci']


def start(args: dict):
	"""
	Starts one of the processes based on command line arguments.

	:return : the worker class that was started
	"""
	config_type = args["type"]

	if args['cmd'] == 'uci':
		disable(999999)  # plz don't interfere with uci

	config = Config(config_type=config_type)
	config.resource.create_directories()
	setup_logger(config.resource.main_log_path)

	logger.info(f"config type: {config_type}")

	if args["cmd"] == 'self':
		from worker import self_play
		return self_play.start(config)
	elif args["cmd"] == 'opt':
		from worker import optimize
		return optimize.start(config)
	elif args["cmd"] == 'eval':
		from worker import evaluate
		return evaluate.start(config)
	elif args["cmd"] == 'sl':
		from worker import sl
		return sl.start(config)
	elif args["cmd"] == 'uci':
		from play_game import uci
		return uci.start(config)
	else:
		raise ValueError(f"{args['cmd']}")
