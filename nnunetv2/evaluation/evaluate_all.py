import os
import subprocess

from nnunetv2.paths import nnUNet_raw, nnUNet_results
from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name

for dataset_id in [27]:
    dataset_name = maybe_convert_to_dataset_name(dataset_id)
    for config in ["2d"]:
        for num_cases in [8, 12, 16, 24, 32, 48, 64, 80, 96, 144, 160, 192, 240]:
            gt_folder = os.path.join(nnUNet_raw, dataset_name, "labelsTs")
            pred_folder = os.path.join(
                nnUNet_results, dataset_name, f"imagesTs_pred_{config}", f"num_training_cases_{num_cases:03d}"
            )

            dataset_file = os.path.join(pred_folder, "dataset.json")
            plans_file = os.path.join(pred_folder, "plans.json")
            args = [gt_folder, pred_folder, "-pfile", plans_file, "-djfile", dataset_file]
            try:
                p = subprocess.Popen(f"python evaluate_predictions.py {' '.join(args)}", shell=True)
                out, err = p.communicate()
                print(out)
            except:
                print(f"Failed for pred folder: {pred_folder}. Does the run/fold exist?")
                continue

# TODO:
# folder_name = "None"
# args = ["-d", dataset_id, "-f", 0, "-c", config, "-i", "imagesTs", "-o", f"imagesTs_pred_192"]
# predict_entry_point(args)
