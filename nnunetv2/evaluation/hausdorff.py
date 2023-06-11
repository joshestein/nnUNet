import numpy as np
import torch
from scipy.spatial.distance import directed_hausdorff
from statistics import mean


def get_symmetric_hausdorff_per_class(prediction: torch.tensor, target: torch.tensor):
    """Calculates the symmetric HD between two tensors. Expects the input tensors to be 4D or 5D, with the second
    dimension corresponding to the number of output classes. Returns the average HD between the two tensors for each
    class, as an array.
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

        if len(prediction.shape) == 4:
            # When we have 2D data, add a dummy depth dimension to avoid conditional behaviour
            prediction = prediction.unsqueeze(2)
            target_onehot = target_onehot.unsqueeze(2)

        for region_index in range(prediction.shape[1]):
            region_distance = []
            for prediction_item, target_item in zip(prediction[:, region_index], target_onehot[:, region_index]):
                for slice_index in range(prediction_item.shape[0]):
                    hd = max(
                        directed_hausdorff(prediction_item[slice_index], target_item[slice_index])[0],
                        directed_hausdorff(target_item[slice_index], prediction_item[slice_index])[0],
                    )
                    region_distance.append(hd)

            hausdorff_distances[region_index] = mean(region_distance)

    return hausdorff_distances
