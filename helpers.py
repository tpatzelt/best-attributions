import matplotlib.pyplot as plt
import numpy as np
import thermostat
import torch
import onnxruntime as ort
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
    return model

def load_albert_v2_traced():
    model = torch.jit.load("models/traced_albert_v2.pt")
    return model

def wrap_call_with_numpy(model):
    """Wraps a pytorch model so that
    single numpy observation can be passed
    and the logits are returned.
    """
    return lambda x: model(torch.tensor(x[None]))[0].detach().numpy()[0]

def load_distilbert():
    ort_session = ort.InferenceSession("models/distilbert-base-uncased-imdb/model-optimized-quantized.onnx")
    callable_expr = lambda x: ort_session.run(["output_0"], dict(input_ids=x[None],
                                                        attention_mask=np.ones_like(x[None])))[0][0]
    return callable_expr


def load_imdb_albert_lig_data():
    return thermostat.load("imdb-albert-lig")


def extract_token_ids_and_attributions(instance, trim_zeros=True, start=0, end=-1):
    tokens = instance.input_ids[start:end]
    attributions = instance.attributions[start:end]
    if trim_zeros:
        tokens = np.trim_zeros(tokens, "b")
        attributions = attributions[:len(tokens)]
    return np.array(tokens), np.array(attributions)
