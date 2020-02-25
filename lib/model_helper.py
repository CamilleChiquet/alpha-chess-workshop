"""
Helper methods for working with trained models.
"""

from logging import getLogger

logger = getLogger(__name__)


def load_best_model_weight(model):
	"""
	:param chess_zero.agent.model.ChessModel model:
	:return:
	"""
	return model.load(model.config.resource.model_best_config_path, model.config.resource.model_best_weight_path)


def save_as_best_model(model):
	"""

	:param chess_zero.agent.model.ChessModel model:
	:return:
	"""
	return model.save(model.config.resource.model_best_config_path, model.config.resource.model_best_weight_path)
