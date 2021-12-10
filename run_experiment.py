import json
from pathlib import Path

from sacred import Experiment
from sacred.observers import MongoObserver, FileStorageObserver

from attribution_methods import RandomAttributionValues
from baselines import ZeroBaselineFactory
from cfg import DB_URI, DB_NAME
from evaluators import ProportionalityEvaluator
from experiment_runner import ExperimentRunner
from models import load_distilbert

ex = Experiment()
ex.observers.extend([MongoObserver(url=DB_URI, db_name=DB_NAME), FileStorageObserver("./logs")])


@ex.config
def base():
    dataset_path = None
    model = None
    attribution_method = {
        "name": None
    }
    num_samples = None
    baseline = None
    try:
        name = "-".join([attribution_method["name"], model, Path(dataset_path).stem, str(num_samples), baseline])
    except TypeError:
        raise TypeError("run experiment with with option.")


@ex.named_config
def random_values():
    dataset_path = "data/imdb-distilbert-1000.json"
    model = "distilbert-quantized"
    attribution_method = {
        "name": "random-attribution-values"
    }
    num_samples = 1
    baseline = "zero"


@ex.automain
def run_experiment(name: str, dataset_path: str, model: str, attribution_method: dict, num_samples: int, baseline: str):
    with open(dataset_path, "r") as fp:
        dataset = json.load(fp)

    if model == "distilbert-quantized":
        model = load_distilbert()
    else:
        raise ValueError(f"Model string '{model}' not supported.")

    if baseline == "zero":
        baseline_factory = ZeroBaselineFactory
    else:
        raise ValueError(f"Baseline string '{baseline}' not supported.")

    evaluator = ProportionalityEvaluator(model=model, baseline_factory=baseline_factory)

    if attribution_method["name"] == "random-attribution-values":
        attribution_method = RandomAttributionValues()

    runner = ExperimentRunner(name=name,
                              num_samples=num_samples,
                              attribution_method=attribution_method,
                              dataset=dataset,
                              evaluator=evaluator,
                              experiment=ex)
    runner.run()
