from dataclasses import dataclass

SAMPLE_REGIONS = ("apex", "mid", "base")


@dataclass(frozen=True)
class SliceRemover:
    percentage_slices: float = 1.0
    sample_regions: list[str] = SAMPLE_REGIONS
    randomise_slices: bool = True
    maintain_shape: bool = True

    def __post_init__(self):
        for region in self.sample_regions:
            assert (
                region in SAMPLE_REGIONS
            ), f"Invalid area {region}. Must be one of {', '.join(f'`{region}`' for region in SAMPLE_REGIONS)}."
