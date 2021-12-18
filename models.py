from typing import Callable

import numpy as np
import onnxruntime as ort
import torch
from scipy.special import softmax
from torchvision.models.vgg import vgg16
from transformers import AutoModelForSequenceClassification


def load_albert_v2() -> torch.nn.Module:
    """Load pytorch model of ALBERT-base-v2 trained on imdb."""
    model = AutoModelForSequenceClassification.from_pretrained("textattack/albert-base-v2-imdb",
                                                               return_dict=False)
    model.eval()
    return model


def load_albert_v2_traced():
    """Load jit model of ALBERT-base-v2 trained on imdb."""
    model = torch.jit.load("models/traced_albert_v2.pt")
    return model


def wrap_pytorch_call_with_numpy(model):
    """Wraps a pytorch model so that
    single numpy observation can be passed
    and the logits are returned.
    """
    return lambda x: model(torch.tensor(x[None]))[0].detach().numpy()[0]


def load_distilbert(return_softmax=1, from_notebook=0) -> Callable:
    """Loads distilbert-base-uncased-imdb from ONNX file. Before saving the model,
    jits optimization and quantifiation routines were executed."""
    model_path = ('../' if from_notebook
                  else '') + "models/distilbert-base-uncased-imdb/model-optimized-quantized.onnx"
    ort_session = ort.InferenceSession(model_path)
    if return_softmax:
        return lambda x: softmax(ort_session.run(["output_0"],
                                                 dict(input_ids=x[None],
                                                      attention_mask=np.ones_like(x[None])))[0][0],
                                 axis=-1)
    else:
        return lambda x: ort_session.run(["output_0"],
                                         dict(input_ids=x[None],
                                              attention_mask=np.ones_like(x[None])))[0][0]


def load_vgg16():
    model = vgg16(pretrained=True).eval()
    return lambda x: model(torch.tensor(x[None], dtype=torch.double))
