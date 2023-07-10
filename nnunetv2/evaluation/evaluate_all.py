import subprocess
from batchgenerators.utilities.file_and_folder_operations import join

for dataset_id in [27]:
    for config in ["2d"]:
        for num_cases in [8, 12, 16, 24, 32, 48, 64, 80, 96, 144, 160, 192, 240]:
            pred_folder = join(f"imagesTs_pred_{config}", f"num_training_cases_{num_cases:03d}")
            args = [
                "-d",
                str(dataset_id),
                "-pred_folder",
                pred_folder,
            ]
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
