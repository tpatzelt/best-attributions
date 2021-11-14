import json

import numpy as np
from tqdm import tqdm

from baselines import ZeroBaselineFactory
from evaluators import ProportionalityEvaluator
from helpers import (load_imdb_distilbert_feature_ablation, load_distilbert)

model = load_distilbert(return_softmax=True)
data = load_imdb_distilbert_feature_ablation()

evaluations = list()
evaluator = ProportionalityEvaluator(model=model, baseline_factory=ZeroBaselineFactory)

for sample in tqdm(data):
    attribution_values = np.array(sample['attributions'])
    observation = np.array(sample['input_ids'])

    tpn = evaluator.compute_tpn(observation=observation,
                                attribution_values=attribution_values)
    tps = evaluator.compute_tps(observation=observation,
                                attribution_values=attribution_values)
    result = {
        'tpn': tpn,
        'tps': tps
    }
    sample.update(result)
    evaluations.append(sample)

with open('data/albert-imdb-feature-ablation-softmax-1000-with-tpn.json', 'w') as fp:
    json.dump(evaluations, fp)
