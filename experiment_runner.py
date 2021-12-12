import json
import os
import time
from typing import List

import numpy as np
from sacred import Experiment
from scipy.special import softmax
from tqdm import tqdm

from attribution_methods import AttributionMethod
from cfg import DATA_PATH
from evaluators import Evaluator


class ExperimentRunner:

    def __init__(self, name: str,
                 num_samples: int,
                 attribution_method: AttributionMethod,
                 dataset: List[dict],
                 evaluator: Evaluator,
                 softmax_attributions: bool,
                 experiment: Experiment = None):
        self.name = name
        self.num_samples = num_samples
        self.attribution_method = attribution_method
        self.dataset = dataset
        self.evaluator = evaluator
        self.experiment = experiment
        self.softmax_attributions = softmax_attributions

    def run(self):
        attributions = []
        for sample in tqdm(self.dataset[:self.num_samples]):
            observation = np.asarray(sample["input_ids"][:512])
            attribution = self.attribution_method.get_attribution_values(observation=observation)
            if self.softmax_attributions:
                attribution = softmax(attribution)
            attributions.append(dict(idx=sample['idx'], attribution_values=attribution.tolist()))

            result = self.evaluator.evaluate(observation=observation,
                                             attribution_values=attribution)
            if self.experiment:
                for name, value in result.items():
                    self.experiment.log_scalar(name=name, value=value)

        attribution_path = str(os.path.join(DATA_PATH, f"{time.strftime('%Y-%d-%m_%H%M')}-"
                                                       f"{self.name}.json"))
        with open(attribution_path, "w") as fp:
            json.dump(attributions, fp)

        if self.experiment:
            self.experiment.add_artifact(attribution_path)
