# %%
import json

from attribution_methods import HillClimber
from baselines import ZeroBaselineFactory
from evaluators import ProportionalityEvaluator
from helpers import load_imdb_albert_lig_data, extract_token_ids_and_attributions, load_distilbert

# %%

data = load_imdb_albert_lig_data()
model = load_distilbert()

# %%

observations, lig_attributions = zip(*[extract_token_ids_and_attributions(d) for d in data[:200]])
del data

evaluator = ProportionalityEvaluator(model=model, baseline_factory=ZeroBaselineFactory)
tpn_objective = lambda x: evaluator.compute_tpn(observation=observation, attribution_values=x)
tps_objective = lambda x: evaluator.compute_tps(observation=observation, attribution_values=x)

# %%
bounds = (0, 1)
iterations = 6
step_size = .01
hill_climber_tpn = HillClimber(objective=tpn_objective, bounds=bounds, iterations=iterations,
                               step_size=step_size)
hill_climber_tps = HillClimber(objective=tps_objective, bounds=bounds, iterations=iterations,
                               step_size=step_size)

## Running Hillclimbing repeatedly on one sample

# %%

iterations = 2
random_idx = 42
observation = observations[random_idx]

tpn_results, tps_results = [], []
for _ in range(iterations):
    break
    hc_tpn_result, _, _ = hill_climber_tpn.get_attribution_values(observation=observation)
    tpn_results.append(
        evaluator.compute_tpn(observation=observation, attribution_values=hc_tpn_result))
    tps_results.append(
        evaluator.compute_tps(observation=observation,attribution_values=hc_tpn_result))

# %%

with open("data/tpn_hillclimber_one_sample.json", "w") as fp:
    json.dump(dict(idx=random_idx,tpn_results=tpn_results, tps_results=tps_results), fp)

# %%

tpn_results, tps_results = [], []
for _ in range(iterations):
    hc_tps_result, _, _ = hill_climber_tps.get_attribution_values(observation=observation)
    tpn_results.append(
        evaluator.compute_tpn(observation=observation, attribution_values=hc_tps_result))
    tps_results.append(
        evaluator.compute_tps(observation=observation,attribution_values=hc_tps_result))

# %%

with open("data/tps_hillclimber_one_sample.json", "w") as fp:
    json.dump(dict(idx=random_idx,tpn_results=tpn_results, random_tps_results=tps_results), fp)

# %%
