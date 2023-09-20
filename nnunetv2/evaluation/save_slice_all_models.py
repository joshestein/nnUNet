import os
import SimpleITK
from matplotlib import pyplot as plt

from nnunetv2.paths import nnUNet_raw, nnUNet_results
from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name

ALL_REGIONS = [
    ("apex", "mid", "base"),
    ("apex", "mid"),
    ("apex", "base"),
    ("mid", "base"),
    ["apex"],
    ["mid"],
    ["base"],
]


def extract_slice(volume_path: str, slice_index: int):
    volume = SimpleITK.ReadImage(volume_path)
    volume = SimpleITK.GetArrayFromImage(volume)
    return volume[slice_index]


def main(dataset_id: int, patient_dir: str, slice_index: int, show_original=True):
    dataset_name = maybe_convert_to_dataset_name(dataset_id)
    raw_folder = os.path.join(nnUNet_raw, dataset_name, "imagesTs")
    labels_folder = os.path.join(nnUNet_raw, dataset_name, "labelsTs")
    base_output_folder = os.path.join(nnUNet_results, dataset_name, "imagesTs_pred_3d_fullres")

    slice = extract_slice(os.path.join(raw_folder, f"{patient_dir}_0000.nii.gz"), slice_index)
    label = extract_slice(os.path.join(labels_folder, f"{patient_dir}.nii.gz"), slice_index)

    plt.ioff()
    plt.tight_layout()

    cols = len(ALL_REGIONS) + 2 if show_original else 1
    fig, axes = plt.subplots(
        nrows=1,
        ncols=cols,
        gridspec_kw=dict(
            wspace=0.05,
            hspace=0.05,
        ),
        figsize=(cols, 2),
    )

    # Zoom in - these numbers are picked until the image looks good
    for ax in axes:
        ax.set_xlim(20, 160)
        ax.set_ylim(50, 190)

    index = 0
    if show_original:
        axes[index].imshow(slice, cmap="gray")
        axes[index].set_title("Original")
        index += 1

    axes[index].imshow(label)
    axes[index].set_title("GT")
    index += 1

    for slice_regions in ALL_REGIONS:
        slice_folder = os.path.join(base_output_folder, f"slice_regions_{'_'.join(slice_regions)}")
        pred = extract_slice(os.path.join(slice_folder, f"{patient_dir}.nii.gz"), slice_index)
        title = "".join([s[0].upper() for s in slice_regions])
        axes[index].imshow(pred)
        axes[index].set_title(title)
        index += 1

    plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
    all_models_folder = os.path.join(base_output_folder, "all_models")
    os.makedirs(all_models_folder, exist_ok=True)
    plt.savefig(os.path.join(all_models_folder, f"{patient_dir}_slice_{slice_index:02d}.png"))
    plt.close(fig)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", "-d", type=int, required=True)
    parser.add_argument("--patient_dir", "-p", type=int, required=True)

    args = parser.parse_args()
    main(args.dataset, args.patient_dir, args.slice)
