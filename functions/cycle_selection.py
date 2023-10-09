# This script contains functions to select the most appropriate cardiac cycle.
import os
import numpy as np
from collections import defaultdict
from functions.general_utilities import *


def _comp_cnr(roi: np.ndarray, background: np.ndarray) -> float:
    """Compute the contrast-to-noise ratio (CNR) of a region of interest (ROI) and background,
    according to the ratio between the mean difference and variance in signal intensities.

    Args:
        roi (np.ndarray): ROI of the image.
        background (np.ndarray): Background of the image.

    Returns:
        cnr (float): Contrast-to-noise ratio (CNR) of the ROI and background.
    """
    # Exclude the pixels that are outside the imaging plane, only to be applied to the background.
    background = background[background != 0]

    diff = abs(np.mean(roi) - np.mean(background))
    var = np.sqrt(np.std(roi) ** 2 + np.std(background) ** 2)
    cnr = diff / var

    return cnr


def _create_mask(
    echo_image: np.ndarray, seg: np.ndarray, desired_labels: list[int]
) -> np.ndarray:
    """Create a mask of the echo image with the desired labels.

    Args:
        echo_image (np.ndarray): echo image.
        seg (np.ndarray): Segmentation of the image.
        desired_labels (list[int]): List of labels to be included in the mask.

    Returns:
        masked_image (np.ndarray): Masked echo image.
    """
    mask = np.zeros(echo_image.shape, dtype=np.uint8)

    # Define mask with desired_labels.
    for label in desired_labels:
        mask[seg == label] = 255

    # Give all pixels within mask same values as image, otherwise exclude them.
    # 300 is an arbitrary value, as long as it is not a value that can be found in the image.
    masked_image = np.where(mask, echo_image, 300)
    masked_image = masked_image[masked_image != 300]

    return masked_image


def _comp_cnr_all_frames(
    path_to_images: str,
    path_to_segmentations: str,
    files_of_view: list[str],
    flagged_frames: list[int],
) -> list[float]:
    """Compute the contrast-to-noise ratio (CNR) of all frames in the image sequence.

    Args:
        path_to_images (str): Path to the directory containing the echo images.
        path_to_segmentations (str): Path to the directory containing the segmentations.
        files_of_view (list[str]): List of image names of one view.
        frames_to_exclude (list[int]): List of frames to be excluded from the CNR calculation.

    Returns:
        cnr_frames (list[float]): List of CNR values for all frames in the image sequence.
    """
    cnr_frames = []

    for frame_nr, filename in enumerate(files_of_view):
        if frame_nr not in flagged_frames:
            # Define file location and load echo image frame.
            file_location_image = define_path_to_images(path_to_images, filename)
            echo_image = convert_image_to_array(file_location_image)

            # Define file location and load segmentation.
            file_location_seg = os.path.join(path_to_segmentations, filename)
            seg = convert_image_to_array(file_location_seg)

            # Create masks for ROI and background.
            mask_roi = _create_mask(echo_image, seg, [1, 3])  # label specific
            mask_background = _create_mask(echo_image, seg, [2])  # label specific

            # Calculate CNR and add to list.
            cnr = _comp_cnr(mask_roi, mask_background)
            cnr_frames.append(cnr)

        else:
            cnr_frames.append(np.nan)

    return cnr_frames


def _find_es_in_cycle(
    es_points: list[int], ed_points_cycle: list[int], lv_areas: list[float]
) -> list[int]:
    """Find the end-systolic (ES) point in the cardiac cycle.

    Args:
        es_points (list[int]): List of ES points in the image sequence.
        ed_points_cycle (list[int]): List of end-diastolic (ED) points in the cardiac cycle.
        lv_areas (list[float]): List of left ventricular (LV) areas in the image sequence.

    Returns:
        es_point_cycle (list[int]): List of ES points in the cardiac cycle.
    """
    # Find all ES points between the ED points in cycle.
    es_points_cycle = [
        es_point
        for es_point in es_points
        if es_point > ed_points_cycle[0] and es_point < ed_points_cycle[1]
    ]

    # If no ES point is defined in cycle, select one based on lowest LV area value within range.
    if len(es_points_cycle) == 0:
        lv_areas_cycle = lv_areas[ed_points_cycle[0] : ed_points_cycle[1]]
        min_lv_area = min([num for num in lv_areas_cycle if num != 0])

        es_point_cycle = [lv_areas.index(min_lv_area)]

    # If more than 1 ES point is defined in cycle, select the point corresponding to the lowest LV area value within range.
    elif len(es_points_cycle) > 1:
        min_area = min([lv_areas[i] for i in es_points_cycle])
        es_point_cycle = [i for i, area in enumerate(lv_areas) if area == min_area]

    else:
        es_point_cycle = es_points_cycle

    return es_point_cycle


def _count_nr_flagged_frames(
    flagged_frames: list[int], first_frame: int, last_frame: int
) -> int:
    """Count the number of flagged frames in the cardiac cycle.

    Args:
        flagged_frames (list[int]): List of flagged frames in the image sequence.
        first_frame (int): First frame of the cardiac cycle.
        last_frame (int): Last frame of the cardiac cycle.

    Returns:
        nr_flagged_frames (int): Number of flagged frames in the cardiac cycle.
    """
    nr_flagged_frames = sum(
        [first_frame < number < last_frame for number in flagged_frames]
    )

    return nr_flagged_frames


def _give_score_per_criterion(
    my_list: list[float], method: str = "max all"
) -> list[int]:
    """Give a score per criterion based on the values.

    Scoring is based on the CNR ("max all") and the number of frames flagged by single-frame QC and multi-frame QC ("min all").

    Args:
        my_list (list[float]): List of values to be scored.
        method (str): Method to be used for scoring (default: "max all").

    Returns:
        scores (list[int]): List of scores for all cardiac cycles.
    """
    # Round values in list to 1 decimals.
    my_list_rounded = [round(item, 1) for item in my_list]

    # Find the minimum and maximum value in the list.
    min_value = min(my_list_rounded)
    max_value = max(my_list_rounded)

    # Score the values in the list based on the method.
    # 4 points if equal to best reference value, 0 point if equal to worst ref value, 2 points for other values.
    if method == "max all":
        scores = [
            0 if item == min_value else 4 if item == max_value else 2
            for item in my_list_rounded
        ]
    elif method == "min all":
        scores = [
            4 if item == min_value else 0 if item == max_value else 2
            for item in my_list_rounded
        ]
    else: 
        raise ValueError("Method not recognized.")

    return scores


def _find_best_cycle(
    cnr_cycles: list[float],
    nr_flagged_frames_sf_qc: list[int],
    nr_flagged_frames_mf_qc: list[int],
) -> int:
    """Find the most appropriate cardiac cycle based on the CNR and the number of flagged frames.

    Args:
        cnr_cycles (list[float]): List of CNR values for all cardiac cycles.
        nr_flagged_frames_sf_qc (list[int]): List of number of flagged frames, selected by single-frame QC, for all cardiac cycles.
        nr_flagged_frames_mf_qc (list[int]): List of number of flagged frames, selected by multi-frame QC, for all cardiac cycles.

    Returns:
        idx_best_cycle (int): Index of the most appropriate cardiac cycle.
    """
    # Calculate scores for each criterion.
    scores_cnr = _give_score_per_criterion(cnr_cycles, "max all")
    scores_sf_qc = _give_score_per_criterion(nr_flagged_frames_sf_qc, "min all")
    scores_mf_qc = _give_score_per_criterion(nr_flagged_frames_mf_qc, "min all")

    # Calculate total score for each cycle.
    scores_tot = [x + y + z for x, y, z in zip(scores_cnr, scores_sf_qc, scores_mf_qc)]

    # Find cycles with the maximum score.
    max_score = max(scores_tot)
    max_score_indices = [
        idx for idx, score_val in enumerate(scores_tot) if score_val == max_score
    ]

    # If more than 1 cycle has the highest score, add 1 to score of cycle with highest cnr.
    if len(max_score_indices) > 1:
        # Check if there is at least 1 cycle with a valid CNR.
        if np.nanmean(cnr_cycles) > 0:
            idx_max_cnr = max(max_score_indices, key=lambda idx: cnr_cycles[idx])
            scores_tot[idx_max_cnr] += 1

        else:
            # If no cycle has a valid CNR, select the cycle with the lowest number of flagged frames.
            idx_min_flagged_frames = min(
                max_score_indices,
                key=lambda idx: nr_flagged_frames_sf_qc[idx]
                + nr_flagged_frames_mf_qc[idx],
            )
            scores_tot[idx_min_flagged_frames] += 1

    # Select the cycle with the highest total score.
    idx_best_cycle = scores_tot.index(max(scores_tot))

    return idx_best_cycle


def _get_properties_best_cycle(
    cnr_frames: list[float],
    ed_points: list[int],
    es_points: list[int],
    lv_areas: list[float],
    flagged_frames_sf_qc: list[int],
    flagged_frames_mf_qc_lv: list[int],
    flagged_frames_mf_qc_la: list[int],
) -> tuple[list[int], list[int]]:
    """Get the end-diastolic (ED) and end-systolic (ES) points of the most appropriate cardiac cycle.

    Args:
        cnr_frames (list[float]): List of contrast-to-noise (CNR) values for all frames in the image sequence.
        ed_points (list[int]): List of ED points in the image sequence.
        es_points (list[int]): List of ES points in the image sequence.
        lv_areas (list[float]): List of left ventricular (LV) areas in the image sequence.
        flagged_frames_sf_qc (list[int]): List of frames flagged by single-frame QC, to be excluded from the CNR calculation.
        flagged_frames_lv_mf_qc (list[int]): List of frames flagged by multi-frame QC for the LV, to be excluded from the CNR calculation.
        flagged_frames_la_mf_qc (list[int]): List of frames flagged by multi-frame QC for the LA, to be excluded from the CNR calculation.

    Returns:
        ed_selected (list[int]): List of ED points in the most appropriate cardiac cycle.
        es_selected (list[int]): List of ES points in the most appropriate cardiac cycle.
    """
    (
        cnr_cycles,
        ed_points_cycles,
        es_points_cycles,
        nr_flagged_frames_sf_qc_cycles,
        nr_flagged_frames_mf_qc_cycles,
    ) = ([], [], [], [], [])

    for idx in range(len(ed_points) - 1):
        ed_idx1, ed_idx2 = ed_points[idx], ed_points[idx + 1]

        # Calculate average CNR of cycle.
        cnr_all = np.nanmean(cnr_frames[ed_idx1 : ed_idx2 + 1])
        cnr_cycles.append(cnr_all)

        # Find ED points in cycle.
        ed_points_cycles.append([ed_idx1, ed_idx2])

        # Find ES point in cycle.
        es_point = _find_es_in_cycle(es_points, [ed_idx1, ed_idx2], lv_areas)
        es_points_cycles.append(es_point[0])

        # Count number of flagged frames single-frame QC.
        count_flagged_frames_sf_qc = _count_nr_flagged_frames(
            flagged_frames_sf_qc, ed_idx1, ed_idx2
        )
        nr_flagged_frames_sf_qc_cycles.append(count_flagged_frames_sf_qc)

        # Count number of flagged frames multi-frame QC.
        count_flagged_frames_lv_mf_qc = _count_nr_flagged_frames(
            flagged_frames_mf_qc_lv, ed_idx1, ed_idx2
        )
        count_flagged_frames_la_mf_qc = _count_nr_flagged_frames(
            flagged_frames_mf_qc_la, ed_idx1, ed_idx2
        )
        nr_flagged_frames_mf_qc_cycles.append(
            (count_flagged_frames_lv_mf_qc + count_flagged_frames_la_mf_qc)
        )

    # Find the ED and ES points of the most appropriate cardiac cycle.
    idx_best_cycle = _find_best_cycle(
        cnr_cycles, nr_flagged_frames_sf_qc_cycles, nr_flagged_frames_mf_qc_cycles
    )

    ed_selected = ed_points_cycles[idx_best_cycle]
    es_selected = _find_es_in_cycle(es_points_cycles, ed_selected, lv_areas)

    return ed_selected, es_selected


def main_cycle_selection(
    path_to_images: str,
    path_to_segmentations: str,
    segmentation_properties: dict[str, dict[str, list[int]]],
    single_frame_qc: dict[str, dict[str, list[int]]],
    multi_frame_qc: dict[str, dict[str, list[int]]],
    all_files: list[str],
    views: list[str],
) -> dict[str, dict[str, list[int]]]:
    """MAIN: Select the most appropriate cardiac cycle from an image sequence containing multiple cycles.

    Args:
        path_to_images (str): Path to the directory containing the echo images.
        path_to_segmentations (str): Path to the directory containing the segmentations.
        segmentation_properties (dict[str, dict[str, list[int]]]): Dictionary containing the segmentation properties.
        single_frame_qc (dict[str, dict[str, list[int]]]): Dictionary containing the results of the single-frame QC.
        multi_frame_qc (dict[str, dict[str, list[int]]]): Dictionary containing the results of the multi-frame QC.
        all_files (list[str]): List of all files in the directory.
        views (list[str]): List of views of the segmentations.

    Returns:
        cycle_info (dict[str, dict[str, list[int]]]): Dictionary containing the end-diastolic (ED) and end-systolic (ES) points of the most appropriate cardiac cycle.
    """
    cycle_info = defaultdict(dict)

    for view in views:
        # Get all files of one view of one person.
        files_of_view = get_list_with_files_of_view(all_files, view)

        # Get the ED and ES points as well as LV areas for a certain patient.
        ed_points = segmentation_properties["ed_points"][view]
        es_points = segmentation_properties["es_points"][view]
        lv_areas = segmentation_properties["lv_areas"][view]

        # Find frames flagged by single-frame QC.
        flagged_frames_sf_qc = single_frame_qc["flagged_frames"][view]

        # Find frames flagged by multi-frame QC.
        flagged_frames_mf_qc_lv = multi_frame_qc["flagged_frames_lv"][view]
        flagged_frames_mf_qc_la = multi_frame_qc["flagged_frames_la"][view]

        # Combine all flagged frames in one list.
        flagged_frames_combined = list(
            set(flagged_frames_sf_qc)
            | set(flagged_frames_mf_qc_lv)
            | set(flagged_frames_mf_qc_la)
        )

        # Calculate the CNR for every frame in an image, if images are present.
        images_present = len(os.listdir(path_to_images)) > 0

        if images_present:
            cnr_frames = _comp_cnr_all_frames(
                path_to_images,
                path_to_segmentations,
                files_of_view,
                flagged_frames_combined,
            )
        else:
            cnr_frames = [0] * len(files_of_view)

        # Get the ED and ES points of most appropriate cardiac cycle.
        ed_selected, es_selected = _get_properties_best_cycle(
            cnr_frames,
            ed_points,
            es_points,
            lv_areas,
            flagged_frames_sf_qc,
            flagged_frames_mf_qc_lv,
            flagged_frames_mf_qc_la,
        )

        # Store the information in a dictionary.
        cycle_info["ed_points_selected"][view] = ed_selected
        cycle_info["es_point_selected"][view] = es_selected
        cycle_info["flagged_frames_combined"][view] = flagged_frames_combined

    return cycle_info
