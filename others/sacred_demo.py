import numpy as np
from sacred import Experiment
from sacred.observers import MongoObserver

ex = Experiment('random_sampling')
ex.observers.append(MongoObserver())


@ex.config
def my_config():
    sample_size = 10


@ex.automain
def my_main(_run, sample_size):
    values = []
    for v in np.random.rand(sample_size):
        values.append(v)
        _run.log_scalar("rand() rolling mean", np.mean(values))
