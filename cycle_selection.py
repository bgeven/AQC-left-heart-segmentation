# This script contains functions to select the most appropriate cardiac cycle.
import os
import numpy as np
from collections import defaultdict
from general_utilities import convert_image


def calculate_cnr(roi, background):
    """Function to calculate the contrast-to-noise ratio (CNR) of a region of interest (ROI) and background,
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


def create_mask(echo_image, seg, desired_labels):
    """Function to create a mask of the US image with the desired labels.

    Args:
        echo_image (np.ndarray): echo image.
        seg (np.ndarray): Segmentation of the image.
        desired_labels (list): List of labels to be included in the mask.

    Returns:
        masked_image (np.ndarray): Masked echo image.
    """
    mask = np.zeros(echo_image.shape, dtype=np.uint8)

    # Define mask with desired_labels.
    for label in desired_labels:
        mask[seg == label] = 255

    # Give all pixels within mask same values as image, otherwise exclude them.
    masked_image = np.where(mask, echo_image, 300)
    masked_image = masked_image[masked_image != 300]

    return masked_image


def get_cnr_all_frames(
    path_to_images, path_to_segmentations, images_of_one_view, frames_to_exclude
):
    """Function to calculate the contrast-to-noise ratio (CNR) of all frames in the image sequence.

    Args:
        path_to_images (str): Path to the directory containing the echo image frames.
        path_to_segmentations (str): Path to the directory containing the segmentations.
        images_of_one_view (list): List of image names of one view.
        frames_to_exclude (list): List of frames to be excluded from the CNR calculation.

    Returns:
        cnr_frames (list): List of CNR values for all frames in the image sequence.
    """
    cnr_frames = []

    for frame_nr, name in enumerate(images_of_one_view):
        if frame_nr not in frames_to_exclude:
            # Define file location and load echo image frame
            file_location_image = os.path.join(
                path_to_images, (name[:-7] + "_0000" + name[-7:])
            )
            echo_image = convert_image(file_location_image)

            # Define file location and load segmentation.
            file_location_seg = os.path.join(path_to_segmentations, name)
            seg = convert_image(file_location_seg)

            # Create masks for ROI and background.
            mask_roi = create_mask(echo_image, seg, [1, 3])  # label specific
            mask_background = create_mask(echo_image, seg, [2])  # label specific

            # Calculate CNR and add to list.
            cnr = calculate_cnr(mask_roi, mask_background)
            cnr_frames.append(cnr)

        else:
            cnr_frames.append(np.nan)

    return cnr_frames


def find_es_in_cycle(ES_points, ED_points_cycle, LV_areas):
    """Function to find the end-systolic (ES) point in the cardiac cycle.

    Args:
        ES_points (list): List of end-systolic (ES) points in the image sequence.
        ED_points_cycle (list): List of end-diastolic (ED) points in the cardiac cycle.
        LV_areas (list): List of left ventricular (LV) areas in the image sequence.

    Returns:
        ES_point_cycle (list): List of end-systolic (ES) points in the cardiac cycle.
    """
    # Find all ES points between the ED points in cycle.
    ES_points_cycle = [
        ES_point
        for ES_point in ES_points
        if ES_point > ED_points_cycle[0] and ES_point < ED_points_cycle[1]
    ]

    # If no ES point is defined in cycle, select one based on lowest LV area value within range.
    if len(ES_points_cycle) == 0:
        LV_areas_cycle = LV_areas[ED_points_cycle[0] : ED_points_cycle[1]]
        min_LV_area = min([num for num in LV_areas_cycle if num != 0])

        ES_point_cycle = [LV_areas.index(min_LV_area)]

    # If more than 1 ES point is defined in cycle, select the point corresponding to the lowest LV area value within range.
    elif len(ES_points_cycle) > 1:
        min_area = min([LV_areas[i] for i in ES_points_cycle])
        ES_point_cycle = [i for i, area in enumerate(LV_areas) if area == min_area]

    else:
        ES_point_cycle = ES_points_cycle

    return ES_point_cycle


def nr_flagged_frames_in_cycle(flagged_frames, first_frame, last_frame):
    """Function to count the number of flagged frames in the cardiac cycle.

    Args:
        flagged_frames (list): List of flagged frames in the image sequence.
        first_frame (int): First frame of the cardiac cycle.
        last_frame (int): Last frame of the cardiac cycle.

    Returns:
        nr_flagged_frames (int): Number of flagged frames in the cardiac cycle.
    """
    nr_flagged_frames = sum([first_frame < number < last_frame for number in flagged_frames])

    return nr_flagged_frames


def give_score_per_criterion(my_list, method="max all"):
    """ Function to score the values in a list based on the method.

    Scoring is based on the CNR ("max all") and the number of frames flagged by single-frame QC and multi-frame QC ("min all").

    Args:
        my_list (list): List of values to be scored.
        method (str): Method to be used for scoring (default: "max all").

    Returns:
        scores (list): List of scores for all cardiac cycles.
    """
    # Round values in list to 1 decimal.
    my_list_rounded = [round(item, 1) for item in my_list]

    # Find the minimum and maximum value in the list.
    min_value = min(my_list_rounded)
    max_value = max(my_list_rounded)

    # Score the values in the list based on the method.
    # 4 points if equal to best reference value, 0 point if equal to worst ref value, 2 points for other values.
    if method == "max all":
        scores = [
            4 if item == max_value else 0 if item == min_value else 2
            for item in my_list_rounded
        ]
    elif method == "min all":
        scores = [
            4 if item == min_value else 0 if item == max_value else 2
            for item in my_list_rounded
        ]

    return scores


def get_best_cycle(cnr_cycles, nr_of_outliers_sf_qc, nr_of_outliers_mf_qc):
    """Function to select the most appropriate cardiac cycle.

    Args:
        cnr_cycles (list): List of CNR values for all cardiac cycles.
        nr_of_outliers_sf_qc (list): List of number of outliers, selected by single-frame QC, for all cardiac cycles.
        nr_of_outliers_mf_qc (list): List of number of outliers, selected by multi-frame QC, for all cardiac cycles.

    Returns:
        cnr_best_cycle (float): CNR of the most appropriate cardiac cycle.
        score_cycle (int): Score of the most appropriate cardiac cycle.
    """
    # Calculate scores for each criterion.
    scores_cnr = give_score_per_criterion(cnr_cycles, "max all")
    scores_sf_qc = give_score_per_criterion(nr_of_outliers_sf_qc, "min all")
    scores_mf_qc = give_score_per_criterion(nr_of_outliers_mf_qc, "min all")

    # Calculate total score for each cycle.
    scores_tot = [x + y + z for x, y, z in zip(scores_cnr, scores_sf_qc, scores_mf_qc)]

    # If more than 1 cycle has the highest score, add 1 to score of cycle with highest cnr.
    max_score = max(scores_tot)
    if scores_tot.count(max_score) > 1:
        idx_list = [
            idx for idx, score_val in enumerate(scores_tot) if score_val == max_score
        ]
        idx_max_cnr = max(idx_list, key=lambda idx: cnr_cycles[idx])
        scores_tot[idx_max_cnr] += 1

    # Select the cycle with the highest total score.
    cnr_best_cycle = cnr_cycles[scores_tot.index(max(scores_tot))]

    score_cycle = max(scores_tot)

    return cnr_best_cycle, score_cycle


def get_main_cycle(
    cnr_frames,
    ed_points,
    es_points,
    lv_areas,
    frames_to_exclude_sf_qc,
    frames_to_exclude_lv_mf_qc,
    frames_to_exclude_la_mf_qc,
):
    """Function to select the most appropriate cardiac cycle.

    Args:
        cnr_frames (list): List of CNR values for all frames in the image sequence.
        ed_points (list): List of end-diastolic (ED) points in the image sequence.
        es_points (list): List of end-systolic (ES) points in the image sequence.
        lv_areas (list): List of left ventricular (LV) areas in the image sequence.
        frames_to_exclude_sf_qc (list): List of frames to be excluded from the CNR calculation, selected by single-frame QC.
        frames_to_exclude_lv_mf_qc (list): List of frames to be excluded from the CNR calculation, selected by multi-frame QC for the left ventricle.
        frames_to_exclude_la_mf_qc (list): List of frames to be excluded from the CNR calculation, selected by multi-frame QC for the left atrium.

    Returns:
        cnr_best_cycle (float): CNR of the most appropriate cardiac cycle.
        ed_selected (list): List of end-diastolic (ED) points in the most appropriate cardiac cycle.
        es_selected (list): List of end-systolic (ES) points in the most appropriate cardiac cycle.
        score (int): Score of the most appropriate cardiac cycle.
    """
    (
        cnr_cycles,
        ed_points_cycles,
        es_points_cycles,
        nr_outliers_sf_qc_cycles,
        nr_outliers_mf_qc_cycles,
    ) = ([], [], [], [], [])

    for idx in range(len(ed_points) - 1):
        # Calculate average CNR of cycle.
        cnr_all = np.nanmean(cnr_frames[ed_points[idx] : ed_points[idx + 1] + 1])
        cnr_cycles.append(cnr_all)

        # Find ED points in cycle.
        ed_points_idx = [ed_points[idx], ed_points[idx + 1]]
        ed_points_cycles.append(ed_points_idx)

        # Find ES point in cycle.
        es_point = find_es_in_cycle(es_points, ed_points_idx, lv_areas)
        es_points_cycles.append(es_point[0])

        # Count number of outliers single-frame QC.
        count_outliers_sf_qc = nr_flagged_frames_in_cycle(
            frames_to_exclude_sf_qc, ed_points[idx], ed_points[idx + 1]
        )
        nr_outliers_sf_qc_cycles.append(count_outliers_sf_qc)

        # Count number of outliers multi-frame QC.
        count_outliers_lv_mf_qc = nr_flagged_frames_in_cycle(
            frames_to_exclude_lv_mf_qc, ed_points[idx], ed_points[idx + 1]
        )
        count_outliers_la_mf_qc = nr_flagged_frames_in_cycle(
            frames_to_exclude_la_mf_qc, ed_points[idx], ed_points[idx + 1]
        )
        nr_outliers_mf_qc_cycles.append(
            (count_outliers_lv_mf_qc + count_outliers_la_mf_qc)
        )

    # Get CNR of best cycle and corresponding ED and ES points.
    cnr_best_cycle, score = get_best_cycle(
        cnr_cycles, nr_outliers_sf_qc_cycles, nr_outliers_mf_qc_cycles
    )
    selection = cnr_cycles.index(cnr_best_cycle)

    ed_selected = ed_points_cycles[selection]
    es_selected = find_es_in_cycle(es_points_cycles, ed_selected, lv_areas)

    return cnr_best_cycle, ed_selected, es_selected, score


def main_cycle_selection(path_to_images, path_to_segmentations, segmentation_properties, single_frame_qc, multi_frame_qc):
    # Get list of filenames in one folder containing the segmentations.
    all_files = os.listdir(path_to_segmentations)
    patients = sorted(set([i[:29] for i in all_files]))

    cycle_info = defaultdict(dict)

    for patient in patients:
        # Get all images of one person.
        images_of_one_person_unsorted = [i for i in all_files if i.startswith(patient)]
        images_of_one_person = sorted(
            images_of_one_person_unsorted, key=lambda x: int(x[30:-7])
        )

        # Get the ED and ES points as well as LV areas for a certain patient. 
        ed_points = segmentation_properties['ED Points'][patient]
        es_points = segmentation_properties['ES Points'][patient]
        lv_areas = segmentation_properties['LV areas'][patient]
    
        # Find frames flagged by single-frame QC.
        flagged_frames_sf_qc = single_frame_qc['Flagged_frames'][patient]
    
        # Find frames flagged by multi-frame QC.
        flagged_frames_mf_qc_lv = multi_frame_qc['Flagged_frames_lv'][patient]
        flagged_frames_mf_qc_la = multi_frame_qc['Flagged_frames_la'][patient]
    
        # Combine all flagged frames in one list.
        flagged_frames_combined = list(set(flagged_frames_sf_qc) | set(flagged_frames_mf_qc_lv) | set(flagged_frames_mf_qc_la))
    
        # Calculate the CNR for every frame in an image
        cnr_frames = get_cnr_all_frames(path_to_images, path_to_segmentations, images_of_one_person, flagged_frames_combined)

        # Get the most appropriate cardiac cycle.
        _, ed_selected, es_selected, _ = get_main_cycle(cnr_frames, ed_points, es_points, lv_areas, flagged_frames_sf_qc, flagged_frames_mf_qc_lv, flagged_frames_mf_qc_la)
    
        # Store the information in a dictionary.
        cycle_info['ED_points_selected'][patient] = ed_selected
        cycle_info['ES_point_selected'][patient] = es_selected
        
    return cycle_info
