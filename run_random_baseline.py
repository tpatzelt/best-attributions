#%%
from sacred import Experiment
from sacred.observers import MongoObserver

ex = Experiment('random_sampling')
ex.observers.append(MongoObserver())

import numpy as np
from baselines import ZeroBaselineFactory
from evaluators import ProportionalityEvaluator
from helpers import load_albert_v2, load_imdb_albert_lig_data, extract_token_ids_and_attributions, wrap_call_with_numpy
from attribution_methods import RandomAttributionValues
from tqdm import tqdm

data = load_imdb_albert_lig_data()
model = load_albert_v2()
callable_model = wrap_call_with_numpy(model)

observation, attribution_values = extract_token_ids_and_attributions(data[42])

evaluator = ProportionalityEvaluator(model=callable_model, baseline_factory=ZeroBaselineFactory)

#%%

tpn_lig = evaluator.compute_tpn(observation=observation, attribution_values=attribution_values)
tps_lig = evaluator.compute_tps(observation=observation, attribution_values=attribution_values)
print(f"{tpn_lig = }")
print(f"{tps_lig = }")

#%%

tpn_results, tps_results = [], []
for i in tqdm(range(1)):
    random_attribution_values = RandomAttributionValues.get_attribution_values(observation=observation)
    tpn = evaluator.compute_tpn(observation=observation, attribution_values=random_attribution_values)
    tps = evaluator.compute_tps(observation=observation, attribution_values=random_attribution_values)
    tpn_results.append(tpn)
    tps_results.append(tps)



#%%

print(np.mean(tpn_results))
print(np.mean(tps_results))

#%%

data = [[x, y] for (x, y) in zip([1,2,3], [1,2,4])]
table = wandb.Table(data=data, columns = ["x", "y"])
wandb.log({"my_custom_plot_id" : wandb.plot.line(table, "x", "y",
           title="Custom Y vs X Line Plot")})

#%%


