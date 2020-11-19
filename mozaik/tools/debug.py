import pickle

import matplotlib.pyplot as plt

from .misc import load_pickle_crosscompat

def visualize_retinal_cache(data_store, which):
    # TODO: f1 is undefined
    cached_stimulus = pickle.load(f1)
    z = load_pickle_crosscompat("./retinal_cache/" + str(which) + ".st")

    pos = data_store.get_neuron_postions()["X_ON"]

    plt.figure()
    plt.scatter(pos[0], pos[1], color=cached_stimulus[:, 0])
