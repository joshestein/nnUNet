import os
from itertools import chain

from nnunetv2.inference.predict_all import run_subprocess
from nnunetv2.paths import nnUNet_raw, nnUNet_results
from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name


def build_args(pred_folder: str, dataset_name: str):
    gt_folder = os.path.join(nnUNet_raw, dataset_name, "labelsTs")
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


def percentage_slices_generator(output_folder: str):
    return (
        os.path.join(output_folder, f"percentage_slices_{percentage_slices}")
        for percentage_slices in [1.0, 0.8, 0.66, 0.5, 0.33, 0.2, 0.1, 0.05]
    )


def main():
    for dataset_id in [27, 114]:
        dataset_name = maybe_convert_to_dataset_name(dataset_id)
        for config in ["2d", "3d_fullres"]:
            base_output_folder = os.path.join(nnUNet_results, dataset_name, f"imagesTs_pred_{config}")

            num_cases_generator = num_training_cases_generator(base_output_folder)
            slice_generator = slice_region_generator(base_output_folder)
            slice_percentage_generator = percentage_slices_generator(base_output_folder)

            for gen_args in chain(num_cases_generator, slice_generator, slice_percentage_generator):
                try:
                    run_subprocess("evaluate_predictions.py", build_args(gen_args, dataset_name))
                except FileNotFoundError:
                    print(f"Failed for {gen_args}. Does the run/fold exist?")
                    continue


if __name__ == "__main__":
    main()
