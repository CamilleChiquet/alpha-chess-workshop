"""
Supervised Learning on generated training data.
"""
import multiprocessing as mp
import os
import sys
import keras.backend as K

_PATH_ = os.path.dirname(os.path.dirname(__file__))

import memory_saving_gradients
"""
Uncomment the following line if your model doesn't fit in your gpu memory.
Training will be a bit slower but you will be able to use bigger networks.
To learn more about : https://medium.com/tensorflow/fitting-larger-networks-into-memory-583e3c758ff9
"""
K.__dict__["gradients"] = memory_saving_gradients.gradients_memory

if _PATH_ not in sys.path:
	sys.path.append(_PATH_)

if __name__ == "__main__":
	mp.set_start_method('spawn')
	sys.setrecursionlimit(10000)
	import manager

	manager.start(worker="opt", config_type="normal")
