import json
from pathlib import Path

from sacred import Experiment
from sacred.observers import MongoObserver, FileStorageObserver

from attribution_methods import RandomAttributionValues, KernelShap, Lime, HillClimber
from baselines import ZeroBaselineFactory
from cfg import DB_URI, DB_NAME
from evaluators import ProportionalityEvaluator, DummyAverageEvaluator
from experiment_runner import ExperimentRunner
from models import load_distilbert

ex = Experiment()
ex.observers.extend([MongoObserver(url=DB_URI, db_name=DB_NAME), FileStorageObserver("./logs")])


@ex.config
def base_config():
    evaluation = {
        "name": None
    }
    dataset = {
        'path': None,
        'num_samples': None
    }
    model = {
        'name': None,
    }
    attribution_method = {
        "name": None
    }
    softmax_attributions = True
    try:
        name = "-".join([str(name) if name else "None" for name in
                         [attribution_method["name"], model["name"], Path(dataset["path"]).stem,
                          evaluation["name"], str(softmax_attributions)]]
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
        'name': 'distilbert',
        'quantized': True
    }
    dataset = {
        'path': "data/imdb-distilbert-1000.json",
        'num_samples': None
    }
    attribution_method = {
        "name": "random-attribution-values",
    }


@ex.named_config
def kernel_shap_config():
    evaluation = {
        "name": "proportionality",
        "baseline_factory": "zero"
    }
    model = {
        'name': 'distilbert',
        'quantized': True
    }
    dataset = {
        'path': "data/imdb-distilbert-1000.json",
        'num_samples': None
    }
    attribution_method = {
        "name": "kernel-shap",
    }


@ex.named_config
def lime_config():
    evaluation = {
        "name": "proportionality",
        "baseline_factory": "zero"
    }
    model = {
        'name': 'distilbert',
        'quantized': True
    }
    dataset = {
        'path': "data/imdb-distilbert-1000.json",
        'num_samples': None
    }
    attribution_method = {
        "name": "lime",
    }


@ex.named_config
def custom_hill_climber_tpn_config():
    evaluation = {
        "name": "proportionality",
        "baseline_factory": "zero"
    }
    model = {
        'name': 'distilbert',
        'quantized': True
    }
    dataset = {
        'path': "data/imdb-distilbert-1000.json",
        'num_samples': 2
    }
    attribution_method = {
        "name": "custom-hill-climber",
        "objective": "tpn",
        "bounds": (.4, .5),
        "iterations": 50,
        "step_size": .1,
    }


@ex.named_config
def custom_hill_climber_tps_config():
    evaluation = {
        "name": "proportionality",
        "baseline_factory": "zero"
    }
    model = {
        'name': 'distilbert',
        'quantized': True
    }
    dataset = {
        'path': "data/imdb-distilbert-1000.json",
        'num_samples': 2
    }
    attribution_method = {
        "name": "custom-hill-climber",
        "objective": "tps",
        "bounds": (.4, .5),
        "iterations": 50,
        "step_size": .1,
    }


@ex.named_config
def dummy_config():
    evaluation = {
        "name": "dummy-average"
    }
    attribution_method = {
        "name": "random-attribution-values",
    }
    dataset = {
        'path': "data/imdb-distilbert-1000.json",
        'num_samples': None
    }


@ex.automain
def run_experiment(name: str, dataset: dict, model: dict, attribution_method: dict, evaluation: dict, softmax_attributions: bool):
    num_samples = dataset["num_samples"]
    with open(dataset["path"], "r") as fp:
        dataset = json.load(fp)

    if model["name"] == "distilbert":
        if model["quantized"]:
            model = load_distilbert()
        else:
            raise ValueError('Only qunatized distilbert available.')

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
        else:
            raise ValueError(f"Hill Climbing Objective string '{attribution_method['objective']}' not supported.")
        attribution_method = HillClimber(bounds=bounds,
                                         iterations=iterations,
                                         step_size=step_size,
                                         objective=objective)

    runner = ExperimentRunner(name=name,
                              num_samples=num_samples,
                              attribution_method=attribution_method,
                              dataset=dataset,
                              evaluator=evaluator,
                              experiment=ex,
                              softmax_attributions=softmax_attributions)
    runner.run()