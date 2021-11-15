#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
from pathlib import Path

sys.path.append(str(Path.cwd().parent))

import json
from sacred import Experiment
from sacred.observers import MongoObserver, FileStorageObserver
from cfg import DB_URI, DB_NAME
from baselines import ZeroBaselineFactory
from evaluators import ProportionalityEvaluator
from helpers import load_imdb_albert_lig_data, extract_token_ids_and_attributions, load_distilbert
from attribution_methods import RandomAttributionValues
from tqdm import tqdm

ex = Experiment('random-baseline-vs-lig', interactive=True)
ex.observers.append(MongoObserver(url=DB_URI, db_name=DB_NAME))
ex.observers.append(FileStorageObserver("logs"))


def _log_scores(name: str, scores: []):
    for score in sorted(scores):
        ex.log_scalar(name, score)


@ex.automain
def run():
    data = load_imdb_albert_lig_data()
    model = load_distilbert()

    observations, lig_attributions = zip(
        *[extract_token_ids_and_attributions(d) for d in data[:1000]])
    del data

    evaluator = ProportionalityEvaluator(model=model, baseline_factory=ZeroBaselineFactory)

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

    with open("data/random_all_samples.json", "w") as fp:
        json.dump(dict(tpn_results=tpn_results, tps_results=tps_results), fp)

    for lig, random in tpn_results:
        ex.log_scalar("lig-tpn-all", lig)
        ex.log_scalar('random-tpn-all', random)
    for lig, random in tps_results:
        ex.log_scalar("lig-tps-all", lig)
        ex.log_scalar('random-tps-all', random)

    iterations = 1000
    random_index = 1
    observation = observations[random_index]
    lig_tpn, lig_tps = tpn_results[random_index][0], tps_results[random_index][0]
    random_attributions = [RandomAttributionValues.get_attribution_values(observation) for _ in
                           range(iterations)]
    random_tpn_results, random_tps_results = [], []
    for attribution in tqdm(random_attributions):
        random_tpn_results.append(
            evaluator.compute_tpn(observation=observation, attribution_values=attribution))
        random_tps_results.append(
            evaluator.compute_tps(observation=observation, attribution_values=attribution))


    with open("data/random_one_sample.json", "w") as fp:
        json.dump(dict(random_tpn_results=random_tpn_results, random_tps_results=random_tps_results,
                       lig_tpn=lig_tpn, lig_tps=lig_tps),
                  fp)
    _log_scores(name="random-tpn-one", scores=random_tpn_results)
    _log_scores(name="random-tps-one", scores=random_tps_results)
    _log_scores(name="lig-tpn-one", scores=[lig_tpn] * len(random_tpn_results))
    _log_scores(name="lig-tps-one", scores=[lig_tps] * len(random_tps_results))

    ### PART 2
    model = load_distilbert(return_softmax=1)
    data = load_imdb_albert_lig_data()

    observations, lig_attributions = zip(
        *[extract_token_ids_and_attributions(d) for d in data[:1000]])
    del data

    evaluator = ProportionalityEvaluator(model=model, baseline_factory=ZeroBaselineFactory)

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

        if i == 2:
            break
    with open("data/random_all_samples-softmax.json", "w") as fp:
        json.dump(dict(tpn_results=tpn_results, tps_results=tps_results), fp)

    # In[ ]:
    for lig, random in tpn_results:
        ex.log_scalar("lig-tpn-all-softmax", lig)
        ex.log_scalar('random-tpn-all-softmax', random)
    for lig, random in tps_results:
        ex.log_scalar("lig-tps-all-softmax", lig)
        ex.log_scalar('random-tps-all-softmax', random)

    observation = observations[random_index]
    lig_tpn, lig_tps = tpn_results[random_index], tps_results[random_index]
    random_attributions = [RandomAttributionValues.get_attribution_values(observation) for _ in
                           range(iterations)]
    random_tpn_results, random_tps_results = [], []
    for attribution in tqdm(random_attributions):
        random_tpn_results.append(
            evaluator.compute_tpn(observation=observation, attribution_values=attribution))
        random_tps_results.append(
            evaluator.compute_tps(observation=observation, attribution_values=attribution))

    # In[ ]:

    with open("data/random_one_sample-softmax.json", "w") as fp:
        json.dump(dict(random_tpn_results=random_tpn_results, random_tps_results=random_tps_results,
                       lig_tpn=lig_tpn, lig_tps=lig_tps),
                  fp)
    _log_scores(name="random-tpn-one-softmax", scores=random_tpn_results)
    _log_scores(name="random-tps-one-softmax", scores=random_tps_results)
    _log_scores(name="lig-tpn-one-softmax", scores=lig_tpn * len(random_tpn_results))
    _log_scores(name="lig-tps-one-softmax", scores=lig_tps * len(random_tpn_results))
