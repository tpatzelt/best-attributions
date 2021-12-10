import matplotlib.pyplot as plt
import numpy as np


def plot_cumulative_values(data):
    idx = range(len(data))
    plt.plot(idx, data)
    cumulative = np.cumsum(data)
    plt.plot(idx, cumulative)


