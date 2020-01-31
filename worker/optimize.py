"""
Encapsulates the worker which trains ChessModels using game data from recorded games from a file.
"""
import os
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from logging import getLogger
from random import shuffle

import numpy as np
import tensorflow as tf
from keras.optimizers import Adam

from agent.model_chess import ChessModel
from config import Config
from env.chess_env import canon_input_planes, is_black_turn
from lib.data_helper import get_game_data_filenames, read_pickle_object, get_next_generation_model_dirs
from lib.model_helper import load_best_model_weight

logger = getLogger(__name__)


def start(config: Config):
	"""
	Helper method which just kicks off the optimization using the specified config
	:param Config config: config to use
	"""
	return OptimizeWorker(config).start()


class OptimizeWorker:
	"""
	Worker which optimizes a ChessModel by training it on game data

	Attributes:
		:ivar Config config: config for this worker
		:ivar ChessModel model: model to train
		:ivar dequeue,dequeue,dequeue dataset: tuple of dequeues where each dequeue contains game states,
			target policy network values (calculated based on visit stats
				for each state during the game), and target value network values (calculated based on
					who actually won the game after that state)
		:ivar ProcessPoolExecutor executor: executor for running all of the training processes
	"""

	def __init__(self, config: Config):
		self.config = config
		self.model = None  # type: ChessModel
		self.dataset = deque(), deque(), deque()
		self.executor = ProcessPoolExecutor(max_workers=config.trainer.cleaning_processes)
		self.lr = self.config.trainer.start_lr

	def start(self):
		"""
		Load the next generation model from disk and start doing the training endlessly.
		"""
		self.model = self.load_model(self.config.trainer.continue_training)
		self.training()

	def training(self):
		"""
		Does the actual training of the model, running it on game data. Endless.
		"""
		if self.config.trainer.tensorboard_enabled:
			self.file_writer = tf.summary.FileWriter(self.config.resource.log_dir + "/" +
		                                         datetime.now().strftime("%Y%m%d-%H%M%S"), tf.Session().graph)

		self.compile_model(lr=self.lr)

		self.filenames = deque(get_game_data_filenames(self.config.resource))
		shuffle(self.filenames)

		all_data_seen_nb_times = 0

		self.epochs = 0
		self.steps_since_last_loss_improvment = 0
		self.last_best_loss = np.inf

		while True:
			self.fill_queue()
			history = self.train_epoch()
			self.tensorboard_logs(history)
			self.update_learning_rate(history)
			self.epochs += 1
			print(f"======= Step : {self.epochs} =======")
			self.save_current_model()
			del self.dataset
			self.dataset = deque(), deque(), deque()
			if len(self.filenames) == 0:
				all_data_seen_nb_times += 1
				self.filenames = deque(get_game_data_filenames(self.config.resource))
				shuffle(self.filenames)
				logger.debug(f"!!! All dataset as been seen {all_data_seen_nb_times} time(s) !!!")

	def update_learning_rate(self, history):
		loss = history.history["loss"][0]

		if loss < self.last_best_loss:
			self.last_best_loss = loss
			self.steps_since_last_loss_improvment = 0
		else:
			self.steps_since_last_loss_improvment += 1
			if self.steps_since_last_loss_improvment >= self.config.trainer.loss_patience:
				self.steps_since_last_loss_improvment = 0
				self.last_best_loss = loss
				self.lr /= 2.0
				if self.lr < self.config.trainer.min_lr:
					self.lr = self.config.trainer.min_lr
				self.compile_model(lr=self.lr)

	def tensorboard_logs(self, history):
		"""
		If tensorboard logs are enabled in the conf (tensorboard_enabled=True), then it will log the loss, policy loss,
		value loss and learning rate evolution across epochs.
		These logs can be viewed on a tensorboard UI by entering the following command in a terminal :
		"tensorboard --logdir=[replace by your logs directory]"
		"""
		if self.config.trainer.tensorboard_enabled:
			loss = history.history["loss"][0]
			summary = tf.Summary(value=[tf.Summary.Value(tag='loss', simple_value=loss)])
			self.file_writer.add_summary(summary, self.epochs)

			policy_out_loss = history.history["policy_out_loss"][0]
			summary = tf.Summary(value=[tf.Summary.Value(tag='policy_out_loss', simple_value=policy_out_loss)])
			self.file_writer.add_summary(summary, self.epochs)

			value_out_loss = history.history["value_out_loss"][0]
			summary = tf.Summary(value=[tf.Summary.Value(tag='value_out_loss', simple_value=value_out_loss)])
			self.file_writer.add_summary(summary, self.epochs)

			summary = tf.Summary(value=[tf.Summary.Value(tag='lr', simple_value=self.lr)])
			self.file_writer.add_summary(summary, self.epochs)

	def train_epoch(self):
		"""
		Runs some number of epochs of training
		:param int epochs: number of epochs
		:return: number of datapoints that were trained on in total

		If you have downloaded all games from 2000 to 2020 with a average elo > 2000 from this site :
		https://www.ficsgames.org/download.html
		Then you have so much data (~1 000 000 games) that you can let validation_split to 0 (default value).
		"""
		tc = self.config.trainer
		state_ary, policy_ary, value_ary = self.collect_all_loaded_data()
		return self.model.model.fit(x=state_ary, y=[policy_ary, value_ary],
		                            batch_size=tc.batch_size,
		                            epochs=1,
		                            shuffle=True,
		                            # validation_split=0.05
		                            verbose=2)

	def compile_model(self, lr: float):
		"""
		Compiles the model to use optimizer and loss function tuned for supervised learning
		"""
		opt = Adam(lr=lr)
		losses = ['categorical_crossentropy', 'mean_squared_error']
		self.model.model.compile(optimizer=opt, loss=losses)

	def save_current_model(self):
		"""
		Saves the current model as the next generation model to the appropriate directory
		"""
		rc = self.config.resource
		model_id = datetime.now().strftime("%Y%m%d-%H%M%S.%f")
		model_dir = os.path.join(rc.next_generation_model_dir, rc.next_generation_model_dirname_tmpl % model_id)
		os.makedirs(model_dir, exist_ok=True)
		config_path = os.path.join(model_dir, rc.next_generation_model_config_filename)
		weight_path = os.path.join(model_dir, rc.next_generation_model_weight_filename)
		self.model.save(config_path, weight_path)

	def fill_queue(self):
		"""
		Fills the self.dataset queues with data from the training dataset.
		"""
		futures = deque()
		with ProcessPoolExecutor(max_workers=self.config.trainer.cleaning_processes) as executor:
			# Loading parallelisation
			for _ in range(self.config.trainer.cleaning_processes):
				if len(self.filenames) == 0:
					break
				filename = self.filenames.popleft()
				logger.debug(f"loading data from {filename}")
				logger.debug(f"\t{len(self.filenames)} files remaining")
				futures.append(executor.submit(load_data_from_file, filename))

			while futures and len(self.dataset[0]) < self.config.trainer.dataset_size:
				for x, y in zip(self.dataset, futures.popleft().result()):
					x.extend(y)
				if len(self.filenames) > 0:
					filename = self.filenames.popleft()
					logger.debug(f"loading data from {filename}")
					logger.debug(f"\t{len(self.filenames)} files remaining")
					futures.append(executor.submit(load_data_from_file, filename))

	def collect_all_loaded_data(self):
		"""

		:return: a tuple containing the data in self.dataset, split into
		(state, policy, and value).
		"""
		state_ary, policy_ary, value_ary = self.dataset

		state_ary1 = np.asarray(state_ary, dtype=np.float32)
		policy_ary1 = np.asarray(policy_ary, dtype=np.float32)
		value_ary1 = np.asarray(value_ary, dtype=np.float32)
		return state_ary1, policy_ary1, value_ary1

	def load_model(self, continue_training: bool):
		"""
		Loads the next generation model from the appropriate directory. If not found, loads
		the best known model.
		"""
		model = ChessModel(self.config)
		rc = self.config.resource

		dirs = get_next_generation_model_dirs(rc)
		if not dirs or len(dirs) == 0 and continue_training:
			logger.debug("loading best model")
			if not load_best_model_weight(model):
				raise RuntimeError("Best model can not loaded!")
		else:
			latest_dir = dirs[-1]
			logger.debug("loading latest model")
			config_path = os.path.join(latest_dir, rc.next_generation_model_config_filename)
			weight_path = os.path.join(latest_dir, rc.next_generation_model_weight_filename)
			model.load(config_path, weight_path, continue_training)
		return model


def load_data_from_file(filename: str):
	return convert_to_cheating_data(fen_data=read_pickle_object(filename + "_fen.pickle"),
	                                moves_data=np.load(filename + "_moves.npy", allow_pickle=True),
	                                scores_data=np.load(filename + "_scores.npy", allow_pickle=True))


def convert_to_cheating_data(fen_data: list, moves_data: np.ndarray, scores_data: np.ndarray):
	state_list = []
	policy_list = []
	value_list = []
	nb_games = len(fen_data)
	for game_index in range(nb_games):
		for move_index in range(len(fen_data[game_index])):
			state_fen, policy, value = fen_data[game_index][move_index], moves_data[game_index][move_index], \
			                           scores_data[game_index][move_index]

			state_planes = canon_input_planes(state_fen)

			if is_black_turn(state_fen):
				policy = Config.flip_policy(policy)

			# TODO : essayer de pondérer sl_value par la position du move dans la partie (petite pondération pour les
			#  premiers et grande pour les derniers, 1 pour le dernier déplacement)
			move_number = int(state_fen.split(' ')[5])
			value_certainty = min(5, move_number) / 5  # reduces the noise of the opening... plz train faster
			sl_value = value * value_certainty + testeval(state_fen, False) * (1 - value_certainty)

			state_list.append(state_planes)
			policy_list.append(policy)
			value_list.append(sl_value)

	return np.asarray(state_list, dtype=np.float32), np.asarray(policy_list, dtype=np.float32), \
	       np.asarray(value_list, dtype=np.float32)
