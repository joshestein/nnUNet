from dataclasses import dataclass

SAMPLE_REGIONS = ("apex", "mid", "base")


@dataclass(frozen=True)
class SliceRemover:
    num_slices: int = None  # The number of slices to sample for each volume. If None, all slices are used.
    sample_regions: list[str] | tuple[str, ...] = SAMPLE_REGIONS
    randomise_slices: bool = True
    maintain_shape: bool = True

    def __post_init__(self):
        for region in self.sample_regions:
            assert (
                region in SAMPLE_REGIONS
            ), f"Invalid area {region}. Must be one of {', '.join(f'`{region}`' for region in SAMPLE_REGIONS)}."
