import json
import time
from typing import List

import numpy as np
from sacred import Experiment
from tqdm import tqdm

from attribution_methods import AttributionMethod
from evaluators import Evaluator


class ExperimentRunner:

    def __init__(self, name: str,
                 num_samples: int,
                 attribution_method: AttributionMethod,
                 dataset: List[dict],
                 evaluator: Evaluator,
                 experiment: Experiment):
        self.name = name
        self.num_samples = num_samples
        self.attribution_method = attribution_method
        self.dataset = dataset
        self.evaluator = evaluator
        self.experiment = experiment

    def run(self):
        attributions = []
        for sample in tqdm(self.dataset[:self.num_samples]):
            observation = np.asarray(sample["input_ids"])
            attribution = self.attribution_method.get_attribution_values(observation=observation)
            attributions.append(dict(idx=sample['idx'], attribution_values=attribution.tolist()))

            result = self.evaluator.evaluate(observation=observation,
                                             attribution_values=attribution)
            for name, value in result.items():
                self.experiment.log_scalar(name=name, value=value)

        attribution_path = f"data/{time.strftime('%Y-%d-%m_%H%M')}-" \
                           f"{self.name}.json"
        with open(attribution_path, "w") as fp:
            json.dump(attributions, fp)

        self.experiment.add_artifact(attribution_path)
