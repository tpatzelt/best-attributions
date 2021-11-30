# %%

import json

import numpy as np
import torch
from captum.attr import FeatureAblation
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

# %%
forward_func = lambda x: torch.tensor([model(np.array(x)[0])])
method = FeatureAblation(forward_func)

# %%

attributions = []
for sample in tqdm(dataset):
    input_ids = torch.tensor([sample["input_ids"]]).long()
    observation = input_ids[0].detach().numpy()
    target_class = torch.tensor(model(observation)).argmax()
    attribution = method.attribute(input_ids, target=target_class, method='gausslegendre')[0] \
        .detach().numpy()
    attribution = attribution * -1 + .2

    tpn = evaluator.compute_tpn(observation=observation,
                                attribution_values=attribution)
    tps = evaluator.compute_tps(observation=observation,
                                attribution_values=attribution)

    attributions.append(dict(idx=sample['idx'], attribution_values=attribution.tolist(),
                             tpn=tpn, tps=tps))

# %%

with open("data/feature-ablation-v2.json", "w") as fp:
    json.dump(attributions, fp)

# %%
