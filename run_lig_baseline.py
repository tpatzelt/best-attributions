import time

import numpy as np
from sacred import Experiment
from sacred.observers import MongoObserver, FileStorageObserver

from baselines import ZeroBaselineFactory
from cfg import DB_URI, DB_NAME
from evaluators import ProportionalityEvaluator
from helpers import (load_albert_v2, load_imdb_albert_lig_data,
                     extract_token_ids_and_attributions,
                     wrap_call_with_numpy, load_albert_v2_traced)

ex = Experiment('scoring-lig')
ex.observers.append(MongoObserver(url=DB_URI, db_name=DB_NAME))
ex.observers.append(FileStorageObserver("logs"))


@ex.config
def config():
    sample_size = None


@ex.automain
def run(sample_size):
    now = time.time()
    data = load_imdb_albert_lig_data()[:sample_size]
    model = load_albert_v2()
    callable_model = wrap_call_with_numpy(model)

    evaluator = ProportionalityEvaluator(model=callable_model, baseline_factory=ZeroBaselineFactory)
    tpn_results, tps_results = [], []
    for i, instance in enumerate(data):
        if i % len(data)/10 == 0:
            print(f"Finished {i}/{len(data)}.")
        observation, attribution_values = extract_token_ids_and_attributions(instance=instance,)
        tpn = evaluator.compute_tpn(observation=observation,
                                    attribution_values=attribution_values)
        tps = evaluator.compute_tps(observation=observation,
                                    attribution_values=attribution_values)
        tpn_results.append(tpn)
        tps_results.append(tps)
    print(f"Finished {len(data)}/{len(data)}.")

    def _log_scores(name: str, scores: []):
        for score in sorted(scores):
            ex.log_scalar(name, score)

    _log_scores(name="tpn", scores=tpn_results)
    _log_scores(name="tps", scores=tps_results)
    _log_scores(name="tpn-sorted", scores=sorted(tpn_results))
    _log_scores(name="tps-sorted", scores=(tpn_results))
    ex.log_scalar("tpn mean over dataset", float(np.mean(tpn_results)))
    ex.log_scalar("tps mean over dataset", float(np.mean(tps_results)))

    print(f"Took {time.time() - now}s.")
