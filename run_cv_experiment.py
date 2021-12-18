import os

import numpy as np
from sacred import Experiment
from sacred.observers import FileStorageObserver, MongoObserver

from attribution_methods import RandomAttributionValues, KernelShap, Lime, HillClimber, NGOpt
from baselines import ZeroBaselineFactory
from cfg import DB_URI, DB_NAME
from dataset import load_imagenet_vgg16_1000
from evaluators import DummyAverageEvaluator
from evaluators_cv import ProportionalityEvaluator
from experiment_runner import ExperimentRunner
from models import load_vgg16

if "MONGO_OBSERVER" in os.environ:
    observers = [
        MongoObserver(url=DB_URI, db_name=DB_NAME),
        FileStorageObserver("./logs")]
else:
    observers = [FileStorageObserver("./logs")]

ex = Experiment()
ex.observers.extend(observers)


@ex.config
def base_config():
    evaluation = {
        "name": None
    }
    dataset = {
        'name': "imagenet_vgg16_1000",
        'num_samples': 500
    }
    model = {
        'name': None,
    }
    attribution_method = {
        "name": None
    }
    try:
        name = "-".join([str(name) if name else "None" for name in
                         [attribution_method["name"], model["name"], dataset["name"],
                          evaluation["name"]]]
                        )
    except TypeError:
        raise TypeError("Experiment cannot run in 'base' mode.")


@ex.named_config
def random_config():
    evaluation = {
        "name": "proportionality",
        "baseline_factory": "zero"
    }
    model = {
        'name': "VGG16",
    }
    attribution_method = {
        "name": "random-attribution-values",
    }


@ex.named_config
def dummy_config():
    evaluation = {
        "name": "dummy-average"
    }
    attribution_method = {
        "name": "random-attribution-values",
    }


@ex.automain
def run_experiment(name: str, dataset: dict,
                   model: dict, attribution_method: dict,
                   evaluation: dict):
    num_samples = dataset["num_samples"]
    if dataset["name"] == "imagenet_vgg16_1000":
        dataset = load_imagenet_vgg16_1000()

    if model["name"] == "VGG16":
        model = load_vgg16()

    elif not model["name"]:
        print("Warning: No model provided for this experiment.")
    else:
        raise ValueError(f"Model string '{model}' not supported.")

    if evaluation["name"] == 'proportionality':
        if evaluation["baseline_factory"] == "zero":
            evaluator = ProportionalityEvaluator(model=model, baseline_factory=ZeroBaselineFactory())
        else:
            raise ValueError("Proportionaly Evaluation only availale with Zero BaselineFactory.")
    elif evaluation["name"] == 'dummy-average':
        evaluator = DummyAverageEvaluator()
    else:
        raise ValueError(f"Evaluation string '{evaluation}' not supported.")

    if attribution_method["name"] == "random-attribution-values":
        attribution_method = RandomAttributionValues()
    elif attribution_method["name"] == "kernel-shap":
        attribution_method = KernelShap(model=model)
    elif attribution_method["name"] == "lime":
        attribution_method = Lime(model=model)
    elif attribution_method["name"] == "custom-hill-climber":
        bounds = attribution_method["bounds"]
        iterations = attribution_method["iterations"]
        step_size = attribution_method["step_size"]
        if attribution_method["objective"] == "tpn":
            objective = lambda observation, candidate: evaluator.compute_tpn(observation=observation, attribution_values=candidate)
        elif attribution_method["objective"] == "tps":
            objective = lambda observation, candidate: evaluator.compute_tps(observation=observation, attribution_values=candidate)
        elif attribution_method["objective"] == "tpn+tps":
            objective1 = lambda observation, candidate: evaluator.compute_tpn(observation=observation, attribution_values=candidate)
            objective2 = lambda observation, candidate: evaluator.compute_tps(observation=observation, attribution_values=candidate)
            objective = lambda observation, candidate: np.mean([objective1(observation, candidate), objective2(observation, candidate)])
        else:
            raise ValueError(f"Hill Climbing Objective string '{attribution_method['objective']}' not supported.")
        attribution_method = HillClimber(bounds=bounds,
                                         iterations=iterations,
                                         step_size=step_size,
                                         objective=objective)
    elif attribution_method['name'] == 'ngopt':
        if attribution_method["objective"] == "tpn":
            objective = lambda observation, candidate: evaluator.compute_tpn(observation=observation, attribution_values=candidate)
        elif attribution_method["objective"] == "tps":
            objective = lambda observation, candidate: evaluator.compute_tps(observation=observation, attribution_values=candidate)
        elif attribution_method["objective"] == "tpn+tps":
            objective1 = lambda observation, candidate: evaluator.compute_tpn(observation=observation, attribution_values=candidate)
            objective2 = lambda observation, candidate: evaluator.compute_tps(observation=observation, attribution_values=candidate)
            objective = lambda observation, candidate: np.mean([objective1(observation, candidate), objective2(observation, candidate)])

        else:
            raise ValueError(f"Hill Climbing Objective string '{attribution_method['objective']}' not supported.")
        attribution_method = NGOpt(
            none_penalty=attribution_method['none_penalty'],
            upper=attribution_method['upper'],
            lower=attribution_method['lower'],
            budget=attribution_method['budget'],
            sigma=attribution_method['sigma'],
            objective=objective
        )

    runner = ExperimentRunner(name=name,
                              num_samples=num_samples,
                              attribution_method=attribution_method,
                              dataset=dataset,
                              evaluator=evaluator,
                              experiment=ex,
                              softmax_attributions=False)
    runner.run()
