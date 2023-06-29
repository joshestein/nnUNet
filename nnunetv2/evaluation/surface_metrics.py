import numpy as np
import surface_distance
import torch


def compute_surface_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    device: torch.device,
    spacing_mm: list[float],
    hausdorff_percentile=95,
):
    if len(prediction.shape) != len(target.shape):
        target = target.view((target.shape[0], 1, *target.shape[1:]))

    if all([i == j for i, j in zip(prediction.shape, target.shape)]):
        # if this is the case, then gt is probably already a one hot encoding
        target_onehot = target
    else:
        target = target.long()
        target_onehot = torch.zeros(prediction.shape, device=device)
        target_onehot.scatter_(1, target, 1)

    prediction = prediction.detach().cpu().numpy()
    target_onehot = target_onehot.detach().cpu().numpy()

    # Ignore background class
    prediction = prediction[:, 1:]
    target_onehot = target_onehot[:, 1:]

    sd = np.empty((prediction.shape[0], prediction.shape[1]))
    hd = np.empty((prediction.shape[0], prediction.shape[1]))

    for batch_index, class_index in np.ndindex(prediction.shape[0], prediction.shape[1]):
        surface_distances = surface_distance.compute_surface_distances(
            # Convert to boolean arrays
            prediction[batch_index, class_index] == 1,
            target_onehot[batch_index, class_index] == 1,
            spacing_mm=spacing_mm,
        )

        hd[batch_index, class_index] = surface_distance.compute_robust_hausdorff(
            surface_distances, hausdorff_percentile
        )
        dist_gt_to_prediction, dist_prediction_to_gt = surface_distance.compute_average_surface_distance(
            surface_distances
        )
        sd[batch_index, class_index] = max(dist_gt_to_prediction, dist_prediction_to_gt)

    # Replace infs with NaNs
    # This allows us to average using np.nanmean
    sd[sd == np.inf] = np.nan
    hd[hd == np.inf] = np.nan

    # Average over batch
    return {
        "avg_surface_distance": np.nanmean(sd, axis=0),
        "hausdorff": np.nanmean(hd, axis=0),
    }
