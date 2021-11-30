# %%
import json

from tqdm import tqdm
import numpy as np
from attribution_methods import HillClimber
from baselines import ZeroBaselineFactory
from evaluators import ProportionalityEvaluator
from helpers import load_imdb_albert_lig_data, extract_token_ids_and_attributions, load_distilbert

# %%

data = load_imdb_albert_lig_data()
model = load_distilbert()

# %%

with open("data/imdb-distilbert-1000.json", "r") as fp:
    dataset = json.load(fp)

evaluator = ProportionalityEvaluator(model=model, baseline_factory=ZeroBaselineFactory)
tpn_objective = lambda x: evaluator.compute_tpn(observation=observation, attribution_values=x)
tps_objective = lambda x: evaluator.compute_tps(observation=observation, attribution_values=x)

# %%
bounds = (.4,.5)
iterations = 50
step_size = .1
hill_climber_tpn = HillClimber(objective=tpn_objective, bounds=bounds, iterations=iterations,
                               step_size=step_size)
hill_climber_tps = HillClimber(objective=tps_objective, bounds=bounds, iterations=iterations,
                               step_size=step_size)

# %% md

## Running Hillclimbing once on all samples

# %%

results = []
for i in tqdm(range(len(dataset))):
    observation = np.array(dataset[i]['input_ids'])
    attributions, _, _ = hill_climber_tpn.get_attribution_values(observation=observation)
    tpn = evaluator.compute_tpn(observation=observation, attribution_values=attributions.copy())
    tps = evaluator.compute_tps(observation=observation,attribution_values=attributions.copy())
    results.append(dict(idx=dataset[i]['idx'], attribution_values=attributions.tolist(),
                             tpn=tpn, tps=tps))

# %%

with open("data/tpn_hillclimber.json", "w") as fp:
    json.dump(results, fp)
# %%

results = []
for i in tqdm(range(len(dataset))):
    observation = np.array(dataset[i]['input_ids'])
    attributions, _, _ = hill_climber_tps.get_attribution_values(observation=observation)
    tpn = evaluator.compute_tpn(observation=observation, attribution_values=attributions.copy())
    tps = evaluator.compute_tps(observation=observation,attribution_values=attributions.copy())
    results.append(dict(idx=dataset[i]['idx'], attribution_values=attributions.tolist(),
                             tpn=tpn, tps=tps))

# %%
with open("data/tps_hillclimber.json", "w") as fp:
    json.dump(results, fp)
