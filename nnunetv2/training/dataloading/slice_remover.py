import math
import numpy as np

SAMPLE_REGIONS = ("apex", "mid", "base")


class SliceRemover:
    """Remove slices from the data. Expects the number of slices to be the second dimension of the data.

    :param percentage_slices: Percentage of slices to keep from the data.
    :param sample_regions: Which area(s) of the heart to sample from. Can be one or more of 'apex', 'mid', 'base'.
        Defaults to ('apex', 'mid', 'base').
    :param randomise_slices: Whether to remove random slices around the midpoint of each sample area. Defaults to `True`.
    :param maintain_shape: Whether to maintain the shape of the data. If `True`, the removed slices will be filled with
            zeros. Defaults to `True`.
    """

    def __init__(
        self,
        percentage_slices: float = 1.0,
        sample_regions: list[str] = SAMPLE_REGIONS,
        randomise_slices=True,
        maintain_shape=True,
    ):
        self.percentage_slices = percentage_slices

        for region in sample_regions:
            assert (
                region in SAMPLE_REGIONS
            ), f"Invalid area {region}. Must be one of {', '.join(f'`{region}`' for region in SAMPLE_REGIONS)}."
        self.sample_regions = sample_regions

        self.randomise_slices = randomise_slices
        self.maintain_shape = maintain_shape

    def remove_slices(self, data, seg):
        if self.percentage_slices == 1.0 and len(self.sample_regions) == len(SAMPLE_REGIONS):
            return data, seg

        assert data.shape == seg.shape

        mask = self.get_mask(data)

        if self.maintain_shape:
            # We zero out everything except the mask
            data[:, ~mask, ...] = 0.0
            seg[:, ~mask, ...] = 0.0
        else:
            # Keep only the samples from the mask
            data = data[:, mask, ...]
            seg = seg[:, mask, ...]

        return data, seg

    def get_mask(self, data):
        slices = data.shape[1]
        slices_per_region = slices / 3  # We divide the entire volume into 3 regions: base, mid, apex
        num_sample_slices = int(slices * self.percentage_slices)

        region_slices = {
            "base": range(0, int(math.ceil(slices_per_region))),
            "mid": range(int(math.ceil(slices_per_region)), int(math.ceil(2 * slices_per_region))),
            "apex": range(int(math.ceil(2 * slices_per_region)), slices),
        }

        if self.randomise_slices:
            indices_to_sample = [index for region in self.sample_regions for index in region_slices[region]]
            if len(indices_to_sample) < num_sample_slices:
                print("Warning: Not enough slices to sample from. Using all slices in the sample regions.")
                indices = indices_to_sample
            else:
                indices = np.random.choice(indices_to_sample, num_sample_slices, replace=False)
        else:
            samples_per_region = int(num_sample_slices / len(self.sample_regions))
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
