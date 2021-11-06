import matplotlib.pyplot as plt
import numpy as np
import thermostat
import torch
from transformers import AutoModelForSequenceClassification


def plot_cumulative_values(data):
    idx = range(len(data))
    plt.plot(idx, data)
    cumulative = np.cumsum(data)
    plt.plot(idx, cumulative)


def thermostat_instance_to_string(instance):
    return instance.tokenizer.convert_tokens_to_string(
        instance.tokenizer.convert_ids_to_tokens(instance.input_ids, skip_special_tokens=True))


def load_albert_v2():
    model = AutoModelForSequenceClassification.from_pretrained("textattack/albert-base-v2-imdb",
                                                               return_dict=False)
    model.eval()
#    model = torch.jit.load("models/traced_albert_v2.pt")
    return model


def wrap_call_with_numpy(model):
    """Wraps a pytorch model so that
    single numpy observation can be passed
    and the logit are returned.
    """
    return lambda x: model(torch.tensor(x[None]))[0].detach().numpy()[0]


def load_imdb_albert_lig_data():
    return thermostat.load("imdb-albert-lig")


def extract_token_ids_and_attributions(instance, start=0, end=-1):
    tokens = instance.input_ids[start:end]
    attributions = instance.attributions[start:end]
    return np.array(tokens), np.array(attributions)
