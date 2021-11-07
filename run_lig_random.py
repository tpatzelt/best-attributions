import time

from sacred import Experiment
from sacred.observers import MongoObserver, FileStorageObserver

from attribution_methods import RandomAttributionValues
from baselines import ZeroBaselineFactory
from cfg import DB_URI, DB_NAME
from evaluators import ProportionalityEvaluator
from helpers import (load_albert_v2, load_imdb_albert_lig_data,
                     extract_token_ids_and_attributions,
                     wrap_call_with_numpy)

ex = Experiment('scoring-lig-vs-random')
ex.observers.append(MongoObserver(url=DB_URI, db_name=DB_NAME))
ex.observers.append(FileStorageObserver("logs"))


@ex.config
def config():
    sample_size = None
    random_sample_size = 2


@ex.automain
def run(sample_size, random_sample_size):
    def _log_scores(name: str, scores: []):
        for score in scores:
            ex.log_scalar(name, score)

    now = time.time()
    data = load_imdb_albert_lig_data()
    data = [data[10], data[20], data[30]]
    model = load_albert_v2()
    callable_model = wrap_call_with_numpy(model)

    evaluator = ProportionalityEvaluator(model=callable_model, baseline_factory=ZeroBaselineFactory)
    tpn_results, tps_results = [], []
    for i, instance in enumerate(data):
        if i % (len(data) / 10) == 0:
            print(f"Finished {i}/{len(data)}.")
        observation, attribution_values = extract_token_ids_and_attributions(instance=instance,
                                                                             end=20)
        tpn = evaluator.compute_tpn(observation=observation,
                                    attribution_values=attribution_values)
        tps = evaluator.compute_tps(observation=observation,
                                    attribution_values=attribution_values)
        tpn_results.append(tpn)
        tps_results.append(tps)
        random_tpn_results_per_sample, random_tps_results_per_sample = [], []
        for j in range(random_sample_size):
            if j % (random_sample_size / 10) == 0:
                print(f"Finished {j}/{random_sample_size}.")
            attribution_values = RandomAttributionValues.get_attribution_values(
                observation=observation)
            tpn = evaluator.compute_tpn(observation=observation,
                                        attribution_values=attribution_values)
            tps = evaluator.compute_tps(observation=observation,
                                        attribution_values=attribution_values)
            random_tpn_results_per_sample.append(tpn)
            random_tps_results_per_sample.append(tps)

        _log_scores("tpn-lig", tpn_results)
        _log_scores("tps-lig", tps_results)

        _log_scores("tpn-per-sample-random", random_tpn_results_per_sample)
        _log_scores("tps-per-sample-random", random_tps_results_per_sample)

    print(f"Finished {len(data)}/{len(data)}.")

    print(f"Took {time.time() - now}s.")
