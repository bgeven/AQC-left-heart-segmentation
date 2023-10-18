# This script contains general functions used in multiple scripts or main workflow.
import os
import cv2
import json
import numpy as np
import SimpleITK as sitk


def get_list_with_views(
    all_files: list[str], length_view_identifier: int
) -> list[str]:
    """Get list of the views of the segmentations present in one folder.

    Args:
        all_files (list[str]): All files in the directory.
        length_view_identifier (int): Length of the view identifier.

    Returns:
        views (list[str]): Plane views of the segmentations.
    """
    # Get list of the views in one folder containing the segmentations of one patient.
    views = sorted(set([i[:length_view_identifier] for i in all_files]))

    return views


def get_list_with_files_of_view(
    all_files: list[str], view_identifier: str, length_ext: int = 7
) -> list[str]:
    """Get list of the files belonging to a specific view.

    Args:
        all_files (list[str]): All files in the directory.
        view_identifier (str): Identifier of the view.
        length_ext (int): Length of the file extension (default: 7 (.nii.gz)).

    Returns:
        images_of_one_view (list[str]): Files belonging to a specific view.
    """
    images_of_one_view_unsrt = [i for i in all_files if i.startswith(view_identifier)]
    images_of_one_view = sorted(
        images_of_one_view_unsrt,
        key=lambda x: int(x[len(view_identifier) + 1 : -length_ext]),
    )

    return images_of_one_view


def convert_image_to_array(file_location: str) -> np.ndarray:
    """Convert nifti or dicom file to 2D array.

    Args:
        file_location (str): Location of the nifti or dicom file.

    Returns:
        image (np.ndarray): 2D array of the image.
    """
    # Load image and convert to array.
    itk_image = sitk.ReadImage(file_location)
    image_array = sitk.GetArrayFromImage(itk_image)

    # If array has a pseudo 3D shape, remove extra dimension (output nnU-Net v1).
    if len(image_array.shape) > 2:
        image = np.reshape(
            image_array.transpose((1, 2, 0)),
            (image_array.shape[1], image_array.shape[2]),
        )
    else:
        image = image_array

    return image


def separate_segmentation(
    seg: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Separate the LV, MYO and LA from the full segmentation into separate segmentations.

    Args:
        seg (np.ndarray): Segmentation of the echo image.

    Returns:
        seg_bg (np.ndarray): Segmentation with label 0 (background).
        seg_lv (np.ndarray): Segmentation with label 1 (LV).
        seg_myo (np.ndarray): Segmentation with label 2 (MYO).
        seg_la (np.ndarray): Segmentation with label 3 (LA).
    """
    seg_bg = np.where(seg == 0, seg, 1).astype(np.uint8)
    seg_lv = np.where(seg == 1, seg, 0)
    seg_myo = np.where(seg == 2, seg, 0)
    seg_la = np.where(seg == 3, seg, 0)

    return seg_bg, seg_lv, seg_myo, seg_la


def find_contours(seg: np.ndarray, spec: str = "all") -> list[np.ndarray]:
    """Find the contours within the segmentation.

    Args:
        seg (np.ndarray): Segmentation of the echo image.
        spec (str): Specification of the contours to find.

    Returns:
        contours (list[np.ndarray]): Contour(s) of a structure.
    """
    # Define the retrieval modes.
    retrieval_modes = {
        "all": cv2.RETR_LIST,
        "external": cv2.RETR_EXTERNAL,
        "tree": cv2.RETR_TREE,
    }

    # Find contours based on the specified mode.
    if spec in retrieval_modes.keys():
        contours, _ = cv2.findContours(
            seg, retrieval_modes.get(spec, cv2.RETR_LIST), cv2.CHAIN_APPROX_NONE
        )
    else:
        raise ValueError("Invalid specification of contours.")

    return contours


def combine_segmentations(
    segmentations: list[np.ndarray],
    typeOfCombination: str = "difference",
    labels: list[int] = [1, 2, 3],
) -> np.ndarray:
    """Combine the segmentations of the LV, MYO and LA into one segmentation.

    A distinction can be made between the following types of combination:
    - difference: each structure gets other value.
    - difference with overlap: each structure gets other value, but overlap is accounted for.
    - no difference: all structures get same value.

    It is assumed that the labels of the segmentations are as described in the README.

    Args:
        segmentations (list[np.ndarray]): Segmentation(s) of different structures in the echo image.
        typeOfCombination (str): Type of combination of the segmentations (default: difference).
        labels (list[int]): Labels of the segmentations that need to be combined (default: [1, 2, 3]).

    Returns:
        total_seg (np.ndarray): Segmentation of the image with all structures.
    """
    # Give the first structure the value of the first label.
    if np.amax(segmentations[0]) == labels[0]:
        total_seg = segmentations[0].copy()
    else:
        total_seg = segmentations[0].copy() * labels[0]

    # Each structure gets other value.
    if typeOfCombination == "difference":
        for nr, seg in enumerate(segmentations[1:]):
            if seg.max() == labels[nr + 1]:
                total_seg += seg
            else:
                total_seg += seg * labels[nr + 1]

    # Each structure gets other value, but overlap is accounted for.
    elif typeOfCombination == "difference_with_overlap":
        for nr, seg in enumerate(segmentations[1:]):
            if seg.max() == labels[nr + 1]:
                total_seg += seg
            else:
                total_seg += seg * labels[nr + 1]

            if labels[nr + 1] == 2:
                total_seg[total_seg > labels[nr + 1]] = labels[nr]
            elif labels[nr + 1] == 3:
                total_seg[total_seg > labels[nr + 1]] = labels[nr + 1]

    # All structures get same value.
    elif typeOfCombination == "no_difference":
        for seg in segmentations[1:]:
            total_seg += seg

        total_seg[total_seg > 0] = 1

    else:
        raise ValueError("Invalid type of combination.")

    return total_seg


def define_path_to_images(
    path_to_images: str, filename: str, length_ext: int = 7, input_channel: str = "0000"
) -> str:
    """Define the path to the image.

    Args:
        path_to_images (str): Path to the directory containing the echo images.
        filename (str): Filename of the segmentation.
        length_ext (int): Length of the file extension (default: 7 (.nii.gz)).
        input_channel (str): Input channel of the image (default: 0000).

    Returns:
        file_location_image (str): Path to the image.
    """
    file_location_image = os.path.join(
        path_to_images,
        (filename[:-length_ext] + "_" + input_channel + filename[-length_ext:]),
    )

    return file_location_image


def load_atlases(path_to_atlases: str) -> tuple[list[float], list[float]]:
    """Load the atlases.

    Args:
        path_to_atlases (str): Path to the atlases.

    Returns:
        atlas_lv (list[float]): Left ventricular atlas.
        atlas_la (list[float]): Left atrial atlas.
    """
    if len(os.listdir(path_to_atlases)) >= 2:
        with open(os.path.join(path_to_atlases, "atlas_lv.json"), "r") as file:
            atlas_lv = json.load(file)

        with open(os.path.join(path_to_atlases, "atlas_la.json"), "r") as file:
            atlas_la = json.load(file)

    else:
        raise ValueError("No atlases found. Please check the path to the atlases.")

    return atlas_lv, atlas_la


def normalise_list(list_to_normalise: list[float]) -> list[float]:
    """Normalise a list of values to a range of 0 to 1.

    Args:
        list_to_normalise (list[float]): Values to normalise.

    Returns:
        normalised_list (list[float]): Normalised values.
    """
    # Find the minimum and maximum values in the list.
    min_value = min(list_to_normalise)
    max_value = max(list_to_normalise)

    # Find the range of values in the list.
    value_range = max_value - min_value

    # Normalise the list of values.
    normalised_list = [(value - min_value) / value_range for value in list_to_normalise]

    return normalised_list
