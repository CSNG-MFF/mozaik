import pickle

import matplotlib.pyplot as plt


def visualize_retinal_cache(data_store, which):
    f = open("./retinal_cache/" + str(which) + ".st", "r")
    # TODO: f1 is undefined
    cached_stimulus = pickle.load(f1)
    z = pickle.load(f)

    pos = data_store.get_neuron_postions()["X_ON"]

    plt.figure()
    plt.scatter(pos[0], pos[1], color=cached_stimulus[:, 0])
