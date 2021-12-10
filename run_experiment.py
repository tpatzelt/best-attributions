import json
from pathlib import Path

from sacred import Experiment
from sacred.observers import MongoObserver, FileStorageObserver

from attribution_methods import RandomAttributionValues
from baselines import ZeroBaselineFactory
from cfg import DB_URI, DB_NAME
from evaluators import ProportionalityEvaluator, DummyAverageEvaluator
from experiment_runner import ExperimentRunner
from models import load_distilbert

ex = Experiment()
ex.observers.extend([MongoObserver(url=DB_URI, db_name=DB_NAME), FileStorageObserver("./logs")])


@ex.config
def base():
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
    try:
        name = "-".join([str(name) if name else "None" for name in
                         [attribution_method["name"], model["name"], Path(dataset["path"]).stem,
                          evaluation["name"]]]
                        )
    except TypeError:
        raise TypeError("Experiment cannot run in 'base' mode.")


@ex.named_config
def random_attributions():
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
        'num_samples': 1000
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
        "name": "random-attribution-values"
    }
    dataset = {
        'path': "data/imdb-distilbert-1000.json",
        'num_samples': 1000
    }


@ex.automain
def run_experiment(name: str, dataset: dict, model: dict, attribution_method: dict, evaluation: dict):
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

    runner = ExperimentRunner(name=name,
                              num_samples=num_samples,
                              attribution_method=attribution_method,
                              dataset=dataset,
                              evaluator=evaluator,
                              experiment=ex)
    runner.run()
