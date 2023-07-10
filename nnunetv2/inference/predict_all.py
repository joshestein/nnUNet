import subprocess
from batchgenerators.utilities.file_and_folder_operations import join

for dataset_id in [27]:
    for config in ["2d"]:
        for num_cases in [8, 12, 16, 24, 32, 48, 64, 80, 96, 144, 160, 192, 240]:
            output_folder = join(f"imagesTs_pred_{config}", f"num_training_cases_{num_cases:03d}")
            args = [
                "-d",
                str(dataset_id),
                "-f",
                str(0),
                "-c",
                config,
                "-m",
                f"num_training_patients_{num_cases}",
                "-i",
                "imagesTs",
                "-o",
                output_folder,
                "-device",
                "cpu",
            ]
            try:
                p = subprocess.Popen(f"python predict_from_raw_data.py {' '.join(args)}", shell=True)
                out, err = p.communicate()
                print(out)
            except:
                print(f"Failed for num_training_cases_{num_cases}. Does the run/fold exist?")
                continue

# TODO:
# folder_name = "None"
# args = ["-d", dataset_id, "-f", 0, "-c", config, "-i", "imagesTs", "-o", f"imagesTs_pred_192"]
# predict_entry_point(args)
