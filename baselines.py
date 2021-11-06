import numpy as np


class ZeroBaselineFactory:
    baseline_value = 0

    @staticmethod
    def apply(mask: np.array, observation: np.array):
        blended_observation = observation.copy()
        blended_observation[mask] = ZeroBaselineFactory.baseline_value
        return blended_observation
