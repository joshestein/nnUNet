import numpy as np
import os
import random
import shutil
from batchgenerators.utilities.file_and_folder_operations import load_json, save_json
from pathlib import Path

from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
from nnunetv2.paths import nnUNet_preprocessed, nnUNet_raw


def make_out_dirs(dataset_id: int, task_name="ACDC"):
    dataset_name = f"Dataset{dataset_id:03d}_{task_name}"

    out_dir = Path(nnUNet_raw.replace('"', "")) / dataset_name
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(out_dir / "imagesTr", exist_ok=True)
    os.makedirs(out_dir / "labelsTr", exist_ok=True)
    os.makedirs(out_dir / "imagesTs", exist_ok=True)
    os.makedirs(out_dir / "labelsTs", exist_ok=True)

    return out_dir


def copy_files(src_data_folder: Path, out_dir: Path):
    """Copy files from the ACDC dataset to the nnUNet dataset folder. Returns the number of training cases."""
    patients_train = sorted([f for f in (src_data_folder / "training").iterdir() if f.is_dir()])
    patients_test = sorted([f for f in (src_data_folder / "testing").iterdir() if f.is_dir()])

    num_training_cases = 0
    # Copy training files and corresponding labels.
    for patient_dir in patients_train:
        for file in patient_dir.iterdir():
            if file.suffix == ".gz" and "_gt" not in file.name and "_4d" not in file.name:
                # The stem is 'patient.nii', and the suffix is '.gz'.
                # We split the stem and append _0000 to the patient part.
                shutil.copy(file, out_dir / "imagesTr" / f"{file.stem.split('.')[0]}_0000.nii.gz")
                num_training_cases += 1
            elif file.suffix == ".gz" and "_gt" in file.name:
                shutil.copy(file, out_dir / "labelsTr" / file.name.replace("_gt", ""))

    # Copy test files.
    for patient_dir in patients_test:
        for file in patient_dir.iterdir():
            if file.suffix == ".gz" and "_gt" not in file.name and "_4d" not in file.name:
                shutil.copy(file, out_dir / "imagesTs" / f"{file.stem.split('.')[0]}_0000.nii.gz")

    return num_training_cases


def convert_acdc(src_data_folder: str, dataset_id=27):
    out_dir = make_out_dirs(dataset_id=dataset_id)
    num_training_cases = copy_files(Path(src_data_folder), out_dir)

    generate_dataset_json(
        str(out_dir),
        channel_names={
            0: "cineMRI",
        },
        labels={
            "background": 0,
            "RV": 1,
            "MLV": 2,
            "LVC": 3,
        },
        file_ending=".nii.gz",
        num_training_cases=num_training_cases,
    )


def create_custom_splits(src_data_folder: Path, dataset_id: int, num_val_cases: int = 40):
    """Creates two additional splits for validating on only ED or ES frames."""
    existing_splits = os.path.join(nnUNet_preprocessed, f"Dataset{dataset_id:03d}_ACDC", "splits_final.json")
    splits = load_json(existing_splits)

    patients_train = sorted([f for f in (src_data_folder / "training").iterdir() if f.is_dir()])

    patient_info = {}
    for patient_dir in patients_train:
        config = np.loadtxt(patient_dir / "Info.cfg", dtype=str, delimiter=":")
        end_diastole = int(config[0, 1])
        end_systole = int(config[1, 1])
        patient_info[patient_dir.name] = {"ed": end_diastole, "es": end_systole}

    splits.append(create_split("ed", patient_info, num_val_cases))
    splits.append(create_split("es", patient_info, num_val_cases))
    save_json(splits, existing_splits)


def create_split(phase_choice: str, patient_info: dict, num_val_cases: int):
    train, val = [], []
    keys = list(patient_info.keys())
    random.shuffle(keys)
    for i, patient_dir in enumerate(keys):
        ed_frame = f"{patient_dir}_frame{patient_info[patient_dir]['ed']:02d}"
        es_frame = f"{patient_dir}_frame{patient_info[patient_dir]['es']:02d}"
        if i < num_val_cases:
            val.append(ed_frame if phase_choice == "ed" else es_frame)
            train.append(es_frame if phase_choice == "ed" else ed_frame)
        else:
            train.append(ed_frame)
            train.append(es_frame)

    return {"train": train, "val": val}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input_folder",
        type=str,
        help="The downloaded ACDC dataset dir. Should contain extracted 'training' and 'testing' folders.",
    )
    parser.add_argument(
        "-d", "--dataset_id", required=False, type=int, default=27, help="nnU-Net Dataset ID, default: 27"
    )
    parser.add_argument(
        "-s",
        "--custom-splits",
        required=False,
        type=bool,
        default=False,
        help="Create custom splits for validating on ED and ES frames.",
    )
    args = parser.parse_args()
    args.input_folder = Path(args.input_folder)

    if args.custom_splits:
        print("Creating custom splits...")
        create_custom_splits(args.input_folder, args.dataset_id)
    else:
        print("Converting...")
        convert_acdc(args.input_folder, args.dataset_id)

    print("Done!")
