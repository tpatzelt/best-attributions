# %%

import json

import numpy as np
import torch
from attribution_methods import RandomAttributionValues
from tqdm import tqdm

from baselines import ZeroBaselineFactory
from evaluators import ProportionalityEvaluator
from helpers import load_distilbert

# %%

with open("data/imdb-distilbert-1000.json", "r") as fp:
    dataset = json.load(fp)
# %%
model = load_distilbert(from_notebook=0)
evaluator = ProportionalityEvaluator(model=model, baseline_factory=ZeroBaselineFactory)

method = RandomAttributionValues()
# %%

attributions = []
for sample in tqdm(dataset):
    input_ids = np.array(sample["input_ids"])
    attribution = method.get_attribution_values(observation=input_ids)

    tpn = evaluator.compute_tpn(observation=input_ids,
                                attribution_values=attribution.copy())
    tps = evaluator.compute_tps(observation=input_ids,
                                attribution_values=attribution.copy())

    attributions.append(dict(idx=sample['idx'], attribution_values=attribution.tolist(),
                             tpn=tpn, tps=tps))

# %%

with open("data/random.json", "w") as fp:
    json.dump(attributions, fp)

# %%
