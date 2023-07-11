import os

import subprocess


def run_subprocess(script: str, args: list[str]):
    assert script.endswith(".py")
    p = subprocess.Popen(f"python {script} {' '.join(args)}", shell=True)
    out, err = p.communicate()
    print(out)


def build_base_args(dataset_id: int, config: str):
    return [
        "-d",
        str(dataset_id),
        "-f",
        str(0),
        "-c",
        config,
        "-i",
        "imagesTs",
    ]


def evaluate_num_training_cases(args: list[str], output_folder: str):
    for num_cases in [8, 12, 16, 24, 32, 48, 64, 80, 96, 144, 160, 192, 240]:
        args.extend(
            [
                "-m",
                f"num_training_patients_{num_cases}",
                "-o",
                os.path.join(output_folder, f"num_training_cases_{num_cases:03d}"),
            ]
        )
        try:
            run_subprocess("predict_from_raw_data.py", args)
        except FileNotFoundError:
            print(f"Failed for num_training_cases_{num_cases}. Does the run/fold exist?")
            continue


def evaluate_slice_regions(args: list[str], output_folder: str):
    for slice_regions in [
        ("apex", "mid", "base"),
        ("apex", "mid"),
        ("apex", "base"),
        ("mid", "base"),
        ["apex"],
        ["mid"],
        ["base"],
    ]:
        args.extend(
            [
                "-m",
                f"num_training_cases_None/percentage_slices_1.0_regions_{'_'.join(slice_regions)}",
                "-o",
                os.path.join(output_folder, f"slice_regions_{'_'.join(slice_regions)}"),
            ]
        )
        try:
            run_subprocess("predict_from_raw_data.py", args)
        except FileNotFoundError:
            print(f"Failed for regions {slice_regions}. Does the run/fold exist?")
            continue


def evaluate_percentage_slices(args: list[str], output_folder: str):
    for percentage_slices in [1.0, 0.8, 0.66, 0.5, 0.33, 0.2, 0.1, 0.05]:
        args.extend(
            [
                "-m",
                f"percentage_slices_{percentage_slices}",
                "-o",
                os.path.join(output_folder, f"percentage_slices_{percentage_slices}"),
            ]
        )
        try:
            run_subprocess("predict_from_raw_data.py", args)
        except FileNotFoundError:
            print(f"Failed for percentage slices {percentage_slices}. Does the run/fold exist?")
            continue


def main():
    for dataset_id in [27, 114]:
        for config in ["2d", "3d_fullres"]:
            base_args = build_base_args(dataset_id, config)
            base_output_folder = f"imagesTs_pred_{config}"
            evaluate_num_training_cases(base_args, base_output_folder)
            evaluate_slice_regions(base_args, base_output_folder)
            evaluate_percentage_slices(base_args, base_output_folder)


if __name__ == "__main__":
    main()
