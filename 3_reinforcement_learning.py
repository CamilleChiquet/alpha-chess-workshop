"""
Reinforcement Learning phase.

After running the supervised phase, put your best model in "data/model/" and name it
model_best_weight.h5 and model_best_config.json.
"""
import multiprocessing as mp
import os
import sys
import keras.backend as K
import memory_saving_gradients

_PATH_ = os.path.dirname(os.path.dirname(__file__))

if _PATH_ not in sys.path:
    sys.path.append(_PATH_)

if __name__ == "__main__":
    mp.set_start_method('spawn')
    sys.setrecursionlimit(10000)
    import manager

    # Alterning self play, training and evaluation phases
    while True:

        # Generates pgn files by competing best network against itself
        manager.start(worker="self", config_type="normal")

        tmp_gradients = K.__dict__["gradients"]
        K.__dict__["gradients"] = memory_saving_gradients.gradients_memory
        for _ in range(2):
            # Continue training network on most recent pgn files
            manager.start(worker="opt", config_type="normal", rl=True)
        K.__dict__["gradients"] = tmp_gradients
        # Evaluates last network against the best and replace it if stronger
        manager.start(worker="eval", config_type="normal")
