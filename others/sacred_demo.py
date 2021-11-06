import numpy as np
from sacred import Experiment
from sacred.observers import MongoObserver
from cfg import DB_URI, DB_NAME
ex = Experiment('random_sampling-1')


ex.observers.append(MongoObserver(url=DB_URI, db_name=DB_NAME))
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
