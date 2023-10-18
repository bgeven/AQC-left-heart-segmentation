# This script contains functions to flag frames based on their structural information.
import os
import cv2
from collections import defaultdict
from functions.general_utilities import *


def _check_for_surrounded_lv(
    seg_1: np.ndarray,
    seg_2: np.ndarray,
    seg_3: np.ndarray,
    lv_surrounded_threshold: int = 1,
) -> bool:
    """Check if the left ventricle is fully surrounded by the myocardium and left atrium.

    Args:
        seg_1 (np.ndarray): Segmentation of the image with only label 1, LV.
        seg_2 (np.ndarray): Segmentation of the image with only label 2, MYO.
        seg_3 (np.ndarray): Segmentation of the image with only label 3, LA.
        lv_surrounded_threshold (int): Threshold to adjust robustness of the check (default: 1.0).

    Returns:
        not_fully_surrounded (bool): Boolean indicating if the left ventricle is fully surrounded by the myocardium and left atrium.
    """
    # Find the contours of the segmentations.
    contour_1 = find_contours(seg_1, "external")
    contour_2 = find_contours(seg_2, "external")
    contour_3 = find_contours(seg_3, "external")

    # Create lists to store minimum distances between LV and MYO, and MYO and LA.
    distances_12, distances_13 = [], []

    # Check if contours are present for all structures.
    if len(contour_1) == 0 or len(contour_2) == 0 or len(contour_3) == 0:
        not_fully_surrounded = True
        return not_fully_surrounded

    for j in range(len(contour_1[0])):
        # Get the coordinates of the left ventricular contour.
        coordinates_1 = tuple([int(contour_1[0][j][0][0]), int(contour_1[0][j][0][1])])

        # Get the distance between the left ventricular contour and the other contours.
        distance_12 = cv2.pointPolygonTest(contour_2[0], coordinates_1, True)
        distances_12.append(abs(distance_12))

        distance_13 = cv2.pointPolygonTest(contour_3[0], coordinates_1, True)
        distances_13.append(abs(distance_13))

    # Check if both structures neighbour the left ventricle.
    if min(distances_12) <= 1 and min(distances_13) <= 1:
        # Find the minimum distance between the border of contour 1 and both other contours.
        distances = [min(pair) for pair in zip(distances_12, distances_13)]

        # Check if the maximum distance is larger than the threshold.
        not_fully_surrounded = max(distances) > lv_surrounded_threshold

    else:
        not_fully_surrounded = True

    return not_fully_surrounded


def _check_for_cut_off_la(
    seg: np.ndarray,
    contour: list[np.ndarray],
    rows_to_exclude: int = 5,
    la_cut_off_threshold: int = 10,
) -> bool:
    """Check if the left atrium is cut off.

    Args:
        seg (np.ndarray): Segmentation of the echo image.
        contour (list[np.ndarray]): Contour of the left atrium.
        rows_to_exclude (int): Number of rows to exclude from the bottom of the segmentation (default: 5).
        la_cut_off_threshold (int): Threshold to adjust robustness of the check (default: 10).

    Returns:
        la_cut_off (bool): Boolean indicating if the left atrium is cut off.
    """
    # Get the number of rows in the segmentation, excluding the x bottom rows.
    vert_boundary_seg = seg.shape[0] - rows_to_exclude

    # Get the vertical coordinates of the LA.
    coordinates_la_vert = [int(contour[0][j][0][1]) for j in range(len(contour[0]))]

    # Get the bottom row of the LA structure.
    maximum_la_row = max(coordinates_la_vert)

    # Check if the LA extends beyond the vertical boundary of the segmentation.
    if maximum_la_row > vert_boundary_seg:
        # Count the number of times the bottom row of the LA is present in the segmentation.
        count = sum(1 for value in coordinates_la_vert if value == maximum_la_row)

        # Check if there are more than x bottom rows of the LA present in the segmentation.
        la_cut_off = count > la_cut_off_threshold
    else:
        la_cut_off = False

    return la_cut_off


def _flag_frames_structural(
    path_to_segmentations: str,
    files_of_view: list[str],
    lv_surrounded_threshold: int = 3,
) -> tuple[list[int], list[int]]:
    """Flag the frames with structural faults, this is a part of multi-frame QC.

    Args:
        path_to_segmentations (str): Path to the directory containing the segmentations.
        files_of_view (list[str]): Filenames names of one view.
        lv_surrounded_threshold (int): Threshold to adjust robustness of the check (default: 3.0).

    Returns:
        flagged_frames_lv (list[int]): Frames to be flagged by multi-frame QC for the left ventricle.
        flagged_frames_la (list[int]): Frames to be flagged by multi-frame QC for the left atrium.
    """
    # Create lists to store frames that do not meet the QC criteria.
    flagged_frames_lv, flagged_frames_la = [], []

    for frame_nr, filename in enumerate(files_of_view):
        # Define file location and load segmentation
        file_location = os.path.join(path_to_segmentations, filename)
        seg = convert_image_to_array(file_location)

        # Separate segmentation in 3 different segmentations.
        _, seg_1, seg_2, seg_3 = separate_segmentation(seg)

        # Find the LA contours.
        contours_3 = find_contours(seg_3, "external")

        # Check if LV is fully surrounded.
        if _check_for_surrounded_lv(seg_1, seg_2, seg_3, lv_surrounded_threshold):
            flagged_frames_lv.append(frame_nr)

        # Check if LA is cut off.
        if (len(contours_3) == 1) and _check_for_cut_off_la(seg_3, contours_3):
            flagged_frames_la.append(frame_nr)

    return flagged_frames_lv, flagged_frames_la


def main_multi_frame_qc_structural(
    path_to_segmentations: str,
    all_files: list[str],
    views: list[str],
    lv_surrounded_threshold: int = 3,
) -> dict[str, dict[str, list[int]]]:
    """MAIN: Do multi-frame QC assessment based on structural criteria.

    Structural criteria:
        - Left ventricle is fully surrounded by the myocardium and left atrium.
        - Left atrium is not cut off by image plane.

    Args:
        path_to_segmentations (str): Path to the directory containing the segmentations.
        all_files (list[str]): All files in the directory.
        views (list[str]): Plane views of the segmentations.
        lv_surrounded_threshold (int): Threshold to adjust robustness of the check (default: 3).

    Returns:
        multi_frame_qc (dict[str, dict[str, list[float]]]): Dictionary containing the results of the multi-frame QC.
    """
    multi_frame_qc = defaultdict(dict)

    for view in views:
        # Get all files of one view of one person.
        files_of_view = get_list_with_files_of_view(all_files, view)

        # Find frames flagged by multi-frame QC.
        flagged_frames_lv, flagged_frames_la = _flag_frames_structural(
            path_to_segmentations, files_of_view, lv_surrounded_threshold
        )

        # Save the results in a dictionary.
        multi_frame_qc["flagged_frames_lv"][view] = flagged_frames_lv
        multi_frame_qc["flagged_frames_la"][view] = flagged_frames_la

    return multi_frame_qc
