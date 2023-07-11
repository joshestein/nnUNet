import os

from nnunetv2.inference.predict_all import run_subprocess
from nnunetv2.paths import nnUNet_raw, nnUNet_results
from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name


def build_args(pred_folder: str, dataset_name: str):
    gt_folder = os.path.join(nnUNet_raw, dataset_name, "labelsTs")
    dataset_file = os.path.join(pred_folder, "dataset.json")
    plans_file = os.path.join(pred_folder, "plans.json")
    return [gt_folder, pred_folder, "-pfile", plans_file, "-djfile", dataset_file]


def predict_num_training_cases(dataset_name: str, config: str):
    for num_cases in [8, 12, 16, 24, 32, 48, 64, 80, 96, 144, 160, 192, 240]:
        pred_folder = os.path.join(
            nnUNet_results, dataset_name, f"imagesTs_pred_{config}", f"num_training_cases_{num_cases:03d}"
        )
        try:
            run_subprocess("evaluate_predictions.py", build_args(pred_folder, dataset_name))
        except FileNotFoundError:
            print(f"Failed for pred folder: {pred_folder}. Does the run/fold exist?")
            continue


def predict_slice_regions(dataset_name: str, config: str):
    for slice_regions in [
        ("apex", "mid", "base"),
        ("apex", "mid"),
        ("apex", "base"),
        ("mid", "base"),
        ["apex"],
        ["mid"],
        ["base"],
    ]:
        pred_folder = os.path.join(
            nnUNet_results, dataset_name, f"imagesTs_pred_{config}", f"slice_regions_{'_'.join(slice_regions)}"
        )
        try:
            run_subprocess("evaluate_predictions.py", build_args(pred_folder, dataset_name))
        except FileNotFoundError:
            print(f"Failed for pred folder: {pred_folder}. Does the run/fold exist?")
            continue


def predict_percentage_slices(dataset_name: str, config: str):
    for percentage_slices in [1.0, 0.8, 0.66, 0.5, 0.33, 0.2, 0.1, 0.05]:
        pred_folder = os.path.join(
            nnUNet_results, dataset_name, f"imagesTs_pred_{config}", f"percentage_slices_{percentage_slices}"
        )
        try:
            run_subprocess("evaluate_predictions.py", build_args(pred_folder, dataset_name))
        except FileNotFoundError:
            print(f"Failed for pred folder: {pred_folder}. Does the run/fold exist?")
            continue


def main():
    for dataset_id in [27, 114]:
        dataset_name = maybe_convert_to_dataset_name(dataset_id)
        for config in ["2d", "3d_fullres"]:
            predict_num_training_cases(dataset_name, config)
            predict_slice_regions(dataset_name, config)
            predict_percentage_slices(dataset_name, config)


if __name__ == "__main__":
    main()
