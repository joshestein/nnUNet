import os

import subprocess

for dataset_id in [27]:
    for config in ["2d"]:
        for slice_regions in [
            ("apex", "mid", "base"),
            ("apex", "mid"),
            ("apex", "base"),
            ("mid", "base"),
            ["apex"],
            ["mid"],
            ["base"],
        ]:
            output_folder = os.path.join(f"imagesTs_pred_{config}", f"slice_regions_{'_'.join(slice_regions)}")
            args = [
                "-d",
                str(dataset_id),
                "-f",
                str(0),
                "-c",
                config,
                "-m",
                f"num_training_cases_None/percentage_slices_1.0_regions_{'_'.join(slice_regions)}",
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
                print(f"Failed for regions {slice_regions}. Does the run/fold exist?")
                continue

# TODO:
# folder_name = "None"
# args = ["-d", dataset_id, "-f", 0, "-c", config, "-i", "imagesTs", "-o", f"imagesTs_pred_192"]
# predict_entry_point(args)
