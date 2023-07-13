import math
import numpy as np
import wandb
from batchgenerators.transforms.abstract_transforms import AbstractTransform

from nnunetv2.training.dataloading.slice_remover import SAMPLE_REGIONS


class SliceRemoverTransform(AbstractTransform):
    def __init__(
        self,
        num_slices: int = None,
        sample_regions: list[str] = SAMPLE_REGIONS,
        randomise_slices=True,
        keys: list[str] = ("data", "seg"),
    ):
        self.num_slices = num_slices
        self.sample_regions = sample_regions
        self.randomise_slices = randomise_slices
        self.keys = keys

    def __call__(self, **data_dict):
        if self.num_slices is None and len(self.sample_regions) == len(SAMPLE_REGIONS):
            return data_dict

        data = data_dict[self.keys[0]]
        mask = self._get_mask(data)

        wandb.config["total_slices"] = data.shape[2]
        wandb.config["usable_slices"] = mask.sum()

        for key in self.keys:
            data = data_dict[key]
            data[:, :, ~mask, ...] = 0
            data_dict[key] = data

        return data_dict

    def _get_mask(self, data):
        """Expects data to be of shape batch x channels x slices x width x height."""
        assert len(data.shape) == 5, "Data must be of shape batch x channels x slices x width x height."

        slices = data.shape[2]

        # Use all the slices when no fixed number is provided
        if self.num_slices is None:
            self.num_slices = slices

        slices_per_region = int(math.ceil(slices / 3))  # We divide the entire volume into 3 regions: base, mid, apex

        region_slices = {
            "base": range(0, slices_per_region),
            "mid": range(slices_per_region, 2 * slices_per_region),
            "apex": range(2 * slices_per_region, slices),
        }

        if self.randomise_slices:
            indices_to_sample = [index for region in self.sample_regions for index in region_slices[region]]
            if len(indices_to_sample) < self.num_slices:
                print("Warning: Not enough slices to sample from. Using all slices in the sample regions.")
                indices = indices_to_sample
            else:
                indices = np.random.choice(indices_to_sample, self.num_slices, replace=False)
        else:
            samples_per_region = int(self.num_slices / len(self.sample_regions))
            indices = []
            for region in self.sample_regions:
                start = region_slices[region][0]
                end = region_slices[region][-1]

                if end - start < samples_per_region:
                    print(f"Warning: Too few slices in region {region} - using all slices.")
                    indices.extend(region_slices[region])
                else:
                    mid = int((start + end) / 2)
                    start = mid - int(samples_per_region / 2)
                    end = mid + int(samples_per_region / 2)
                    indices.extend(range(start, end))

        mask = np.zeros(slices, dtype=bool)
        mask[indices] = True
        return mask
