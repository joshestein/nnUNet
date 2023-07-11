import os
import subprocess
from itertools import chain


def run_subprocess(script: str, args: list[str]):
    assert script.endswith(".py")
    p = subprocess.Popen(f"python {script} {' '.join(args)}", shell=True)
    out, err = p.communicate()
    print(out)


def build_base_args(dataset_id: int, config: str):
    return ["-d", str(dataset_id), "-f", str(0), "-c", config, "-i", "imagesTs", "-device", "cpu"]


def num_training_cases_generator(output_folder: str):
    return (
        [
            "-m",
            f"num_training_patients_{num_cases}",
            "-o",
            os.path.join(output_folder, f"num_training_cases_{num_cases:03d}"),
        ]
        for num_cases in [8, 12, 16, 24, 32, 48, 64, 80, 96, 144, 160, 192, 240]
    )


def slice_region_generator(output_folder: str):
    return (
        [
            "-m",
            f"num_training_cases_None/percentage_slices_1.0_regions_{'_'.join(slice_regions)}",
            "-o",
            os.path.join(output_folder, f"slice_regions_{'_'.join(slice_regions)}"),
        ]
        for slice_regions in [
            ("apex", "mid", "base"),
            ("apex", "mid"),
            ("apex", "base"),
            ("mid", "base"),
            ["apex"],
            ["mid"],
            ["base"],
        ]
    )


def percentage_slices_generator(output_folder: str):
    return (
        [
            "-m",
            f"percentage_slices_{percentage_slices}",
            "-o",
            os.path.join(output_folder, f"percentage_slices_{percentage_slices}"),
        ]
        for percentage_slices in [1.0, 0.8, 0.66, 0.5, 0.33, 0.2, 0.1, 0.05]
    )


def main():
    for dataset_id in [27, 114]:
        for config in ["2d", "3d_fullres"]:
            base_args = build_base_args(dataset_id, config)
            base_output_folder = f"imagesTs_pred_{config}"

            num_cases_generator = num_training_cases_generator(base_output_folder)
            slice_generator = slice_region_generator(base_output_folder)
            slice_percentage_generator = percentage_slices_generator(base_output_folder)

            for gen_args in chain(
                num_cases_generator,
                slice_generator,
                slice_percentage_generator,
            ):
                try:
                    run_subprocess("predict_from_raw_data.py", base_args + gen_args)
                except FileNotFoundError:
                    print(f"Failed for {gen_args}. Does the run/fold exist?")
                    continue


if __name__ == "__main__":
    main()
