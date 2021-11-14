#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
from pathlib import Path

sys.path.append(str(Path.cwd().parent))

import json
from sacred import Experiment
from sacred.observers import MongoObserver, FileStorageObserver

from baselines import ZeroBaselineFactory
from evaluators import ProportionalityEvaluator
from helpers import load_imdb_albert_lig_data, extract_token_ids_and_attributions, load_distilbert
from attribution_methods import RandomAttributionValues
from tqdm import tqdm

# In[2]:


ex = Experiment('random_baseline', interactive=True)
ex.observers.append([MongoObserver(), FileStorageObserver("../logs")])


def _log_scores(name: str, scores: []):
    for score in sorted(scores):
        ex.log_scalar(name, score)


# In[11]:


data = load_imdb_albert_lig_data()
model = load_distilbert(return_softmax=1)

# In[12]:


observations, lig_attributions = zip(*[extract_token_ids_and_attributions(d) for d in data[:2000]])
del data

evaluator = ProportionalityEvaluator(model=model, baseline_factory=ZeroBaselineFactory)

# In[13]:


tpn_results, tps_results = [], []
for i in tqdm(range(len(observations))):
    observation = observations[i]
    lig_attribution = lig_attributions[i]
    random_attribution = RandomAttributionValues.get_attribution_values(observation=observation)
    tpn_results.append((
        evaluator.compute_tpn(observation=observation, attribution_values=lig_attribution),
        evaluator.compute_tpn(observation=observation, attribution_values=random_attribution)))
    tps_results.append((
        evaluator.compute_tps(observation=observation, attribution_values=lig_attribution),
        evaluator.compute_tps(observation=observation, attribution_values=random_attribution)))

# In[ ]:

with open("../data/random_all_samples.json", "w") as fp:
    json.dump(dict(tpn_results=tpn_results, tps_results=tps_results), fp)

