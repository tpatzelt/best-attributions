# %%

import json

import numpy as np
from tqdm import tqdm

from baselines import ZeroBaselineFactory
from evaluators import ProportionalityEvaluator
from helpers import load_distilbert, load_imdb_albert_lig_data, extract_token_ids_and_attributions

# %%
lig_dataset = load_imdb_albert_lig_data()

with open("data/imdb-distilbert-1000.json", "r") as fp:
    dataset = json.load(fp)
# %%
model = load_distilbert(from_notebook=0)
evaluator = ProportionalityEvaluator(model=model, baseline_factory=ZeroBaselineFactory)

# %%

attributions = []
for sample in tqdm(dataset):
    input_ids = np.array(sample["input_ids"])
    _, attribution = extract_token_ids_and_attributions(lig_dataset[sample['idx']])
    attribution = np.resize(attribution, input_ids.shape)

    tpn = evaluator.compute_tpn(observation=input_ids,
                                attribution_values=attribution.copy())
    tps = evaluator.compute_tps(observation=input_ids,
                                attribution_values=attribution.copy())
    attributions.append(dict(idx=sample['idx'], attribution_values=attribution.tolist(),
                             tpn=tpn, tps=tps))

# %%

with open("data/lig.json", "w") as fp:
    json.dump(attributions, fp)

# %%
