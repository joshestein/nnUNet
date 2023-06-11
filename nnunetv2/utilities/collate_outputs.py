import numpy as np
from typing import List


def collate_outputs(outputs: List[dict]):
    """
    used to collate default train_step and validation_step outputs. If you want something different then you gotta
    extend this

    we expect outputs to be a list of dictionaries where each of the dict has the same set of keys
    """
    collated = {}
    for k in outputs[0].keys():
        if np.isscalar(outputs[0][k]):
            collated[k] = [o[k] for o in outputs]
        elif isinstance(outputs[0][k], np.ndarray):
            collated[k] = np.vstack([o[k][None] for o in outputs])
        elif isinstance(outputs[0][k], list):
            collated[k] = [item for o in outputs for item in o[k]]
        elif isinstance(outputs[0][k], dict):
            collated[k] = {k2: [] for k2 in outputs[0][k].keys()}
            for k2 in outputs[0][k].keys():
                collated[k][k2] = [o[k][k2] for o in outputs]
        else:
            raise ValueError(
                f"Cannot collate input of type {type(outputs[0][k])}. "
                f"Modify collate_outputs to add this functionality"
            )
    return collated
