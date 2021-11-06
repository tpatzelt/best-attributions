import numpy as np


class RandomAttributionValues:
    @staticmethod
    def get_attribution_values(observation: np.array):
        return np.random.rand(observation.size).reshape(observation.shape)

