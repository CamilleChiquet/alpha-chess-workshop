"""
Manages starting off each of the separate processes involved in ChessZero -
self play, training, and evaluation.
"""
from logging import getLogger, disable

from config import Config
from lib.logger import setup_logger

logger = getLogger(__name__)


def start(worker: str, config_type: str = "normal", model_1_path: str = None,
          model_2_path: str = None, deterministic: bool = False):
	"""
	Starts one of the processes based on given arguments.

	:return : the worker class that was started
	"""

	if worker == 'uci':
		disable(999999)  # plz don't interfere with uci

	config = Config(config_type=config_type)
	config.resource.create_directories()
	setup_logger(config.resource.main_log_path)

	logger.info(f"config type: {config_type}")

	if worker == 'opt':
		from worker import optimize
		return optimize.start(config)
	elif worker == 'eval':
		from worker import evaluate
		return evaluate.start(config)
	elif worker == 'sl':
		from worker import sl
		return sl.start(config)
	elif worker == 'uci':
		from play_game import uci
		return uci.start(config)
	elif worker == 'self':
		from worker import self
		return self.start(config)
	elif worker == 'duel':
		from worker import duel
		return duel.start(config, model_1_path=model_1_path, model_2_path=model_2_path, deterministic=deterministic)
	else:
		raise ValueError(f"{worker}")
