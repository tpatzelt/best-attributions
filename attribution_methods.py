import numpy as np
import abc

class AttributionMethod(abc.ABC):
    def get_attribution_values(self, observation: np.array):
        raise NotImplementedError

class RandomAttributionValues(AttributionMethod):
    def get_attribution_values(self, observation: np.array):
        return np.random.random_sample(observation.shape)


class HillClimber(AttributionMethod):
    def __init__(self, objective, bounds, iterations, step_size):
        self.objective = objective
        self.bounds = bounds
        self.iterations = iterations
        self.step_size = step_size

    def get_attribution_values(self, observation: np.array):
        bounds = np.asarray([self.bounds for _ in range(len(observation))])
        # generate an initial point
        solution = bounds[:, 0] + np.random.rand(len(bounds)) * (bounds[:, 1] - bounds[:, 0])
        # evaluate the initial point
        solution_eval = self.objective(solution)
        # run the hill climb
        solutions = list()
        solutions.append(solution)
        for i in range(self.iterations):
            # take a step
            candidate = solution + np.random.randn(len(bounds)) * self.step_size
            # evaluate candidate point
            candidate_eval = self.objective(candidate)
            if candidate_eval is None:
                continue
            # check if we should keep the new point
            if candidate_eval <= solution_eval:
                # store the new point
                solution, solution_eval = candidate, candidate_eval
                # keep track of solutions
                solutions.append(solution)
                # report progress
                # print('>%d = %.5f' % (i, solution_eval))
        return (solution, solution_eval, solutions)
