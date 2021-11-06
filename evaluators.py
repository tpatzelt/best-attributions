from typing import List, Tuple, Generator

import numpy as np
from numpy import ma
from sklearn.metrics import auc
from numpy.typing import NDArray


class ProportionalityEvaluator:
    def __init__(self, baseline_factory, model):
        self.baseline_factory = baseline_factory
        self.model = model

    @staticmethod
    def create_sorted_index(x: np.ndarray) -> np.array:
        """
        Sorts x and returns an index in ascending order.

        Each element is a tuple that corresponds to a location in x.
        x[sorted_index(x)[k] returns the k-th largest value in the array.

        :param x: Multi-dimensional array to be indexed.
        :return: List of indices in ascending order.
        """
        index = np.unravel_index(np.argsort(x, axis=None), x.shape)
        return index[0]

    @staticmethod
    def iterate_masks_saliency_ratio(
            attribution_values: np.array,
            reverse: bool = False,
            saliency_ratio_per_step: float = 0.1) -> Generator[Tuple[np.array, float], None, None]:
        """
            Generates masks that cover a set of pixels according to the ratio of
            the saliency of these pixels compared to the total saliency.

            Returns a Generator that starts from an empty (or full if reverse is true)
                mask, i.e., all mask entries are False (or True), and flips the mask in
                decreasing order of saliency values.

            :param attribution_values:
            :param reverse: Sets the direction of masking,
                False means gradually masking more and more pixels in the image,
                True means the pixels are gradually unmasked
            :param saliency_ratio_per_step: The ratio of saliency of flipped mask entries compared
                to the total sum of saliency, per iteration.

            :yield: Saliency masks.
        """
        masked_saliency_map = ma.array(attribution_values)
        if reverse:
            masked_saliency_map.mask = np.ones_like(attribution_values)

        current_ratio = 0
        smap_rectified = np.maximum(attribution_values, 0)
        total_saliency = np.sum(smap_rectified)
        sorted_index = ProportionalityEvaluator.create_sorted_index(attribution_values)
        for index in sorted_index[::-1]:
            if smap_rectified[index] != 0:
                current_ratio += smap_rectified[index] / total_saliency
                masked_saliency_map[index] = ma.nomask if reverse else ma.masked

            # in the last step, we also return a saliency map to include all pixels
            end = index == sorted_index[::-1][-1]

            if saliency_ratio_per_step < current_ratio or end:
                yield masked_saliency_map.copy(), current_ratio
                current_ratio = 0

    def _ablate_and_predict(self, mask: np.array, observation: np.array,
                            predicted_class: NDArray[int]) -> float:
        ablated_observation = self.baseline_factory.apply(observation=observation, mask=mask)
        return self.model(ablated_observation)[predicted_class]

    def _get_proportionality_value(
            self,
            observation: np.array,
            masks_ratios: List[Tuple[np.array, float]],
    ) -> float:
        """
        Calculates the total proportionality for a set of masks and their ratios of
        pixels covered.

        :param observation: Observation input to the model.
        :param masks_ratios: A list with masks and their corresponding ratio of pixels covered.
        :return: The total proportionality value that results from applying the masks and
            calculating the area under the curve
        """

        original_output = self.model(observation)
        predicted_class = np.argmax(original_output)
        baseline_input = self.baseline_factory.apply(mask=np.ones(observation.shape).astype(bool),
                                                     observation=observation)
        baseline_output = self.model(baseline_input)
        baseline_confidence = baseline_output[predicted_class]

        masks, ratios = zip(*[(m[0].mask, m[1]) for m in masks_ratios])

        proportionality_values = []
        ratio_values = []
        previous_ratio = 0
        last_output_value = 0.0
        for (mask_normal, ratio_normal), (mask_reverse, _) in zip(
                zip(masks, ratios), zip(masks[::-1], ratios[::-1])
        ):
            ablated_prediction_normal = self._ablate_and_predict(mask_normal, observation,
                                                                 predicted_class)
            ablated_prediction_reverse = self._ablate_and_predict(mask_reverse, observation,
                                                                  predicted_class)
            proportionality_value = abs(ablated_prediction_normal - ablated_prediction_reverse)
            proportionality_values.append(proportionality_value)
            ratio_values.append(previous_ratio + ratio_normal)
            previous_ratio += ratio_normal
            last_output_value = ablated_prediction_normal

        normalizing_factor = 1 / (
                original_output[predicted_class] * min(1, baseline_confidence / last_output_value)
        )
        return normalizing_factor * auc(
            x=np.asarray(ratio_values), y=np.asarray(proportionality_values)
        )

    def compute_tpn(self, observation, attribution_values, saliency_ratio_per_step=.2):
        masks_ratios = list(
            ProportionalityEvaluator.iterate_masks_saliency_ratio(
                attribution_values=attribution_values,
                saliency_ratio_per_step=saliency_ratio_per_step))
        tpn_score = self._get_proportionality_value(observation=observation,
                                                    masks_ratios=masks_ratios)
        return tpn_score

    def compute_tps(self, observation, attribution_values, saliency_ratio_per_step=.2):
        masks_ratios = list(
            ProportionalityEvaluator.iterate_masks_saliency_ratio(
                attribution_values=attribution_values,
                reverse=True,
                saliency_ratio_per_step=saliency_ratio_per_step))
        tpn_score = self._get_proportionality_value(observation=observation,
                                                    masks_ratios=masks_ratios)
        return tpn_score
