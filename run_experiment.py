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
    evaluation = "proportionality"
    dataset_path = None
    model = None
    attribution_method = {
        "name": None
    }
    num_samples = None
    baseline = None
    try:
        name = "-".join([str(name) if name else "None" for name in
                         [attribution_method["name"], model, Path(dataset_path).stem,
                          num_samples, baseline, evaluation]]
                        )
    except TypeError:
        raise TypeError("Experiment cannot run in 'base' mode.")


@ex.named_config
def random_attributions():
    dataset_path = "data/imdb-distilbert-1000.json"
    model = "distilbert-quantized"
    attribution_method = {
        "name": "random-attribution-values"
    }
    num_samples = 1
    baseline = "zero"


@ex.named_config
def dummy_config():
    evaluation = "dummy-average"
    dataset_path = "data/imdb-distilbert-1000.json"
    attribution_method = {
        "name": "random-attribution-values"
    }
    num_samples = None
    baseline = "zero"


@ex.automain
def run_experiment(name: str, dataset_path: str, model: str, attribution_method: dict, num_samples: int, baseline: str, evaluation: str):
    with open(dataset_path, "r") as fp:
        dataset = json.load(fp)

    if model == "distilbert-quantized":
        model = load_distilbert()
    elif model is None:
        print("Warning: No model provided for this experiment.")
    else:
        raise ValueError(f"Model string '{model}' not supported.")

    if baseline == "zero":
        baseline_factory = ZeroBaselineFactory
    elif baseline is None:
        print("Warning: No baseline provided for this experiment.")
        baseline_factory = None
    else:
        raise ValueError(f"Baseline string '{baseline}' not supported.")

    if evaluation == 'proportionality':
        evaluator = ProportionalityEvaluator(model=model, baseline_factory=baseline_factory)
    elif evaluation == 'dummy-average':
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
