"""Everything related to loading, parsing, altering, ... a dataset."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_imdb_distilbert_first_1000(max_length: int = 512):
    with open("data/imdb-distilbert-first-1000.json", "r") as fp:
        dataset = json.load(fp)
    for observation in dataset:
        observation['input_ids'] = observation['input_ids'][:max_length]
        observation['attention_mask'] = observation['attention_mask'][:max_length]
        observation['tokens'] = observation['tokens'][:max_length]

        # rename to observation to be consistent through different domains
        observation["observation"] = observation.pop("input_ids")
    return dataset


def load_imagenet_vgg16_1000():
    path = Path("../imagenet-sample-images")
    if not path.exists():
        raise RuntimeError("ImageNet samples not found. Pull from 'https://github.com/EliSchwartz/imagenet-sample-images'.")

    def grey2rgb(img):
        gray_normalized = img.clip(0, 80) / 80 * 255
        return np.stack([gray_normalized] * 3, axis=2)

    def normalize(image):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        if image.max() > 1:
            image /= 255
        if image.ndim == 2:
            image = grey2rgb(image)
        image = image - mean
        image = image / std
        # image = image[None]
        # return image.swapaxes(-1, 1).swapaxes(2, 3)
        return image

    for i, path in enumerate(sorted([path for path in path.iterdir() if path.suffix == ".JPEG"])):
        img = plt.imread(str(path), format="jpeg")
        img = img.astype(float)
        norm_img = normalize(img)  # .astype(np.half)
        yield dict(idx=i, observation=norm_img.tolist())
