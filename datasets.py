"""Everything related to loading, parsing, altering, ... a dataset."""
import json


def load_imdb_distilbert_first_1000(max_length: int = 512):
    with open("data/imdb-distilbert-first-1000.json", "r") as fp:
        dataset = json.load(fp)
    for observation in dataset:
        observation['input_ids'] = observation['input_ids'][:max_length]
        observation['attention_mask'] = observation['attention_mask'][:max_length]
        observation['tokens'] = observation['tokens'][:max_length]

        observation["observation"] = observation.pop("input_ids")
    return dataset
