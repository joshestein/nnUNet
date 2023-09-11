import os
import SimpleITK
from batchgenerators.utilities.file_and_folder_operations import join, subfiles
from itertools import chain
from matplotlib import pyplot as plt

from nnunetv2.evaluation.evaluate_all import (
    integer_slice_region_generator,
    num_training_cases_generator,
    percentage_slices_generator,
    proportion_generator,
    slice_region_generator,
)
from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO
from nnunetv2.paths import nnUNet_raw, nnUNet_results
from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name


def main(ground_truth_folder: str, labels_folder: str, pred_folder: str):
    files_pred = subfiles(labels_folder, suffix=".nii.gz", join=False)
    files_ground_truth = subfiles(ground_truth_folder)

    files_ref = [join(labels_folder, i) for i in files_pred]
    files_pred = [join(pred_folder, i) for i in files_pred]

    slice_dir = os.path.join(pred_folder, "slices")
    os.makedirs(slice_dir, exist_ok=True)

    plt.ioff()
    plt.tight_layout()

    for ground_truth, ref, pred in zip(files_ground_truth, files_ref, files_pred):
        patient = ref.split("/")[-1].split(".")[0]

        img = SimpleITK.ReadImage(ground_truth)
        img = SimpleITK.GetArrayFromImage(img)

        try:
            label, _ = SimpleITKIO().read_seg(ref)
            pred, _ = SimpleITKIO().read_seg(pred)
        except:
            continue

        slices = label.shape[1]
        cols = 3
        fig, axes = plt.subplots(
            nrows=slices,
            ncols=3,
            gridspec_kw=dict(
                wspace=0.0,
                hspace=0.05,
                top=1.0 - 0.5 / (slices + 1),
                bottom=0.5 / (slices + 1),
                left=0.5 / (cols + 1),
                right=1 - 0.5 / (cols + 1),
            ),
            figsize=(cols + 1, slices + 1),
        )

        cols = ["Original", "Label", "Prediction"]
        for ax, col in zip(axes[0], cols):
            ax.set_title(col)

        for i in range(slices):
            axes[i, 0].imshow(img[i])
            axes[i, 1].imshow(label[0, i])
            axes[i, 2].imshow(pred[0, i])

        plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
        plt.savefig(os.path.join(slice_dir, f"{patient}_slices.png"), bbox_inches="tight")
        plt.close(fig)


if __name__ == "__main__":
    for dataset_id in [27, 114]:
        dataset_name = maybe_convert_to_dataset_name(dataset_id)
        gt_folder = os.path.join(nnUNet_raw, dataset_name, "imagesTs")
        gt_labels_folder = os.path.join(nnUNet_raw, dataset_name, "labelsTs")
        base_output_folder = os.path.join(nnUNet_results, dataset_name, "imagesTs_pred_3d_fullres")

        num_cases_generator = num_training_cases_generator(base_output_folder)
        slice_generator = slice_region_generator(base_output_folder)
        slice_percentage_generator = percentage_slices_generator(base_output_folder)
        integer_slices = integer_slice_region_generator(base_output_folder)
        proportion_balance = proportion_generator(dataset_id, base_output_folder)

        for pred_folder in chain(
            num_cases_generator,
            # slice_generator,
            # slice_percentage_generator,
            # integer_slices,
            # proportion_balance,
        ):
            try:
                main(gt_folder, gt_labels_folder, pred_folder)
            except FileNotFoundError:
                print(f"Failed for {pred_folder}. Does the run/fold exist?")
                continue
