import numpy as np
import torch
from scipy.spatial.distance import directed_hausdorff
from statistics import mean


def get_symmetric_hausdorff_per_class(prediction: torch.tensor, target: torch.tensor):
    """Calculates the symmetric HD between two tensors. Expects the prediction to be in a B x N x H x W format,
    where N is the number of dimensions after one-hot encoding. Returns the HD between the two tensors for each
    one-hot dimension, in dictionary form.
    """
    hausdorff_distances = np.empty(prediction.shape[1])

    with torch.no_grad():
        if len(prediction.shape) != len(target.shape):
            target = target.view((target.shape[0], 1, *target.shape[1:]))

        if all([i == j for i, j in zip(prediction.shape, target.shape)]):
            # if this is the case then gt is probably already a one hot encoding
            target_onehot = target
        else:
            target = target.long()
            target_onehot = torch.zeros(prediction.shape, device=prediction.device)
            target_onehot.scatter_(1, target, 1)

        for region_index in range(prediction.shape[1]):
            region_distance = []
            for index, (prediction_item, target_item) in enumerate(zip(prediction, target_onehot)):
                hd = max(
                    directed_hausdorff(prediction_item[region_index], target_item[region_index])[0],
                    directed_hausdorff(target_item[region_index], prediction_item[region_index])[0],
                )
                region_distance.append(hd)

            hausdorff_distances[region_index] = mean(region_distance)

    return hausdorff_distances
