import abc
from concurrent import futures
from typing import Callable

import captum.attr
import nevergrad as ng
import numpy as np
import torch
from captum.attr import KernelShap as CaptumKernelShap, Lime as CaptumLime


class AttributionMethod(abc.ABC):
    def get_attribution_values(self, observation: np.array):
        raise NotImplementedError

class NGOpt(AttributionMethod):
    def __init__(self, objective, none_penalty: float, lower:float, upper:float, sigma:float, budget:int):
        self.objective = objective
        self.none_penalty = none_penalty
        self.lower = lower
        self.upper = upper
        self.sigma = sigma
        self.budget = budget

    def get_attribution_values(self, observation: np.array):
        def objective(candidate):
            res = self.objective(observation, candidate)
            if res is None:
                return 10  # float("inf")
            return res

        param = ng.p.Array(shape=observation.shape)
        param.set_bounds(lower=self.lower, upper=self.upper)
        param.set_mutation(sigma=self.sigma)

        optimizer = ng.optimizers.NGOpt(parametrization=param, budget=self.budget, num_workers=4)
        with futures.ThreadPoolExecutor(max_workers=optimizer.num_workers) as executor:
            recommendation = optimizer.minimize(objective, executor=executor, batch_mode=False)
        return recommendation.value

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
        solution_eval = self.objective(observation, solution)
        # run the hill climb
        solutions = list()
        solutions.append(solution)
        for i in range(self.iterations):
            # take a step
            candidate = solution + np.random.randn(len(bounds)) * self.step_size
            # evaluate candidate point
            candidate_eval = self.objective(observation, candidate)
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
        # return (solution, solution_eval, solutions)
        return solution


class CaptumAttributionMethod(AttributionMethod):
    method: type(captum.attr.Attribution)

    def __init__(self, model: Callable):
        self.model = model
        forward_func = lambda x: torch.tensor(model(x.squeeze().numpy())[None])
        if self.method:
            self.method = self.method(forward_func=forward_func)
        else:
            raise RuntimeError("Don't use CaptumAttributionMethod directly.")

    def get_attribution_values(self, observation: np.array):
        target_class = torch.tensor(np.argmax(self.model(observation)))
        observation = torch.tensor(observation[None]).long()
        attribution = self.method.attribute(observation, target=target_class)
        return attribution[0].detach().numpy()


class KernelShap(CaptumAttributionMethod):
    method = CaptumKernelShap


class Lime(CaptumAttributionMethod):
    method = CaptumLime
