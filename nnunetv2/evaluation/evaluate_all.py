import math
import os
from itertools import chain

from nnunetv2.inference.predict_all import run_subprocess
from nnunetv2.paths import nnUNet_raw, nnUNet_results
from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name


def build_args(gt_folder, pred_folder: str):
    dataset_file = os.path.join(pred_folder, "dataset.json")
    plans_file = os.path.join(pred_folder, "plans.json")
    return [gt_folder, pred_folder, "-pfile", plans_file, "-djfile", dataset_file]


def num_training_cases_generator(output_folder: str):
    return (
        os.path.join(output_folder, f"num_training_cases_{num_cases:03d}")
        for num_cases in [8, 12, 16, 24, 32, 48, 64, 80, 96, 144, 160, 192, 240]
    )


def slice_region_generator(output_folder: str):
    return (
        os.path.join(output_folder, f"slice_regions_{'_'.join(slice_regions)}")
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


def integer_slice_region_generator(output_folder: str):
    return (
        os.path.join(output_folder, f"num_slices_{num_slices}") for num_slices in [1, 2, 4, 5, 6, 8, 10, 13, 14, 16, 20]
    )


def percentage_slices_generator(output_folder: str):
    return (
        os.path.join(output_folder, f"percentage_slices_{percentage_slices}")
        for percentage_slices in [1.0, 0.8, 0.66, 0.5, 0.33, 0.2, 0.1, 0.05]
    )


def proportion_generator(dataset_id: int, output_folder: str):
    proportion_constant = 1360
    if dataset_id == 114:
        num_cases = [240, 192, 160, 144, 120, 100]
    else:
        num_cases = [160, 144, 120, 100, 80, 65]
    slices = [math.ceil(proportion_constant / v) for v in num_cases]

    return (
        os.path.join(output_folder, f"num_training_cases_{num_training_cases:03d}_num_slices_{num_slices}")
        for num_training_cases, num_slices in zip(num_cases, slices)
    )


def main():
    for dataset_id in [27, 114]:
        dataset_name = maybe_convert_to_dataset_name(dataset_id)
        gt_folder = os.path.join(nnUNet_raw, dataset_name, "labelsTs")
        for config in ["2d", "3d_fullres"]:
            base_output_folder = os.path.join(nnUNet_results, dataset_name, f"imagesTs_pred_{config}")

            num_cases_generator = num_training_cases_generator(base_output_folder)
            slice_generator = slice_region_generator(base_output_folder)
            slice_percentage_generator = percentage_slices_generator(base_output_folder)
            integer_slices = integer_slice_region_generator(base_output_folder)
            proportion_balance = proportion_generator(dataset_id, base_output_folder)

            for pred_folder in chain(
                num_cases_generator, slice_generator, slice_percentage_generator, integer_slices, proportion_balance
            ):
                try:
                    run_subprocess("evaluate_predictions.py", build_args(gt_folder, pred_folder))
                except FileNotFoundError:
                    print(f"Failed for {pred_folder}. Does the run/fold exist?")
                    continue


if __name__ == "__main__":
    main()
