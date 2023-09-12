# This script contains functions for the single-frame quality control of segmentation.
import os
import cv2
import numpy as np
from collections import defaultdict
from general_utilities import *


def num_contours(seg):
    """Find the number of external contours in a segmentation.

    Args:
        seg (np.ndarray): Segmentation of the image.

    Returns:
        number_of_contours (int): Number of external contours in the segmentation.
    """
    # Perform morphological closing to fill in small gaps in segmentation.
    closed_seg = cv2.morphologyEx(seg, cv2.MORPH_CLOSE, np.array([2, 2]))

    contours = find_contours(closed_seg, "external")
    number_of_contours = len(contours)

    return number_of_contours


def check_for_gaps(num_of_contours_ext, contours_all, min_size=[2, 2]):
    """Check if gaps are present in a segmentation.

    Args:
        num_of_contours_ext (int): Number of external contours in the segmentation.
        contours_all (list): List of all contours in the segmentation.
        min_size (list): Minimum size of a gap.

    Returns:
        num_of_gaps (int): Number of gaps in the segmentation.
    """
    # Pick contours larger than the defined minimum value.
    filtered_contours = [
        contour
        for contour in contours_all
        if cv2.contourArea(contour) >= (min_size[0] * min_size[1])
    ]
    num_of_contours_all = len(filtered_contours)

    # Check if gaps are present in segmentation.
    num_of_gaps = num_of_contours_all - num_of_contours_ext

    return num_of_gaps


def check_gap_between_structures(seg_A, seg_B, num_gaps_A, num_gaps_B, min_size=[1, 1]):
    """Check if gaps are present between two structures.

    Args:
        seg_A (np.ndarray): Segmentation of the first structure.
        seg_B (np.ndarray): Segmentation of the second structure.
        num_gaps_A (int): Number of gaps in the first structure.
        num_gaps_B (int): Number of gaps in the second structure.
        min_size (list): Minimum size of a gap.

    Returns:
        num_gaps (int): Number of gaps between the two structures.
    """
    # Create total segmentation of different combinations of structures
    total_seg = combine_segmentations([seg_A, seg_B], "no_difference")

    # Find contours within combined segmentations 1 and 2
    contours_AB = find_contours(total_seg, "all")
    filtered_contours = [
        contour
        for contour in contours_AB
        if cv2.contourArea(contour) >= (min_size[0] * min_size[1])
    ]

    # Find number of gaps within segmentations A and B
    nr_of_gaps = num_gaps_A + num_gaps_B

    # Define the presence of a gap between 2 structures as the number of contours seg A and B
    # minus number of gaps within seg A and B minus 1 (because of minimal number of contours A and B of 1).
    num_gaps = len(filtered_contours) - nr_of_gaps - 1

    return num_gaps


def check_for_gaps_full(
    contours, num_gaps_1, num_gaps_2, num_gaps_3, num_gaps_12, num_gaps_13, min_size=[1, 1]
):
    """Check if gaps are present in a segmentation.

    Args:
        contours (list): List of all contours in the segmentation.
        num_gaps_1 (int): Number of gaps in the first structure.
        num_gaps_2 (int): Number of gaps in the second structure.
        num_gaps_3 (int): Number of gaps in the third structure.
        num_gaps_12 (int): Number of gaps between the first and second structure.
        num_gaps_13 (int): Number of gaps between the first and third structure.
        min_size (list): Minimum size of a gap (default: [1, 1]).

    Returns:
        num_other_gaps (int): Number of gaps in the segmentation not belonging to the structures.
    """
    # Pick contours larger than the defined minimum value.
    filtered_contours = [
        contour
        for contour in contours
        if cv2.contourArea(contour) >= (min_size[0] * min_size[1])
    ]

    # Subtract the number of gaps in each structure from the number of contours larger than the defined
    # minimum value, minus 1 added for correction of inclusion of the full contour as 1.
    num_other_gaps = (
        len(filtered_contours)
        - num_gaps_1
        - num_gaps_2
        - num_gaps_3
        - num_gaps_12
        - num_gaps_13
        - 1
    )

    return num_other_gaps


def do_single_frame_qc(directory_segmentations, images_of_one_person, min_gap_size=[2, 2]):
    """Do single-frame quality control of segmentation.

    This quality control is based on the following criteria:
    1. Only one structure of each label is present in segmentation.
    2. No gaps within and between structures.

    Args:
        directory_segmentations (str): Directory of the segmentations.
        images_of_one_person (list): List of all images of one person.
        min_gap_size (list): Minimum size of a gap (default: [2, 2]).

    Returns:
        qc_scores (list): List of QC scores per image.
        overviews (dict): Dictionary of all interim results per image.
        flagged_frames (list): List of frames with a QC score > 0.
    """
    qc_scores, overviews = [], {}

    # Loop over all images of one person
    for image in images_of_one_person:
        # Define file location and load segmentation.
        file_location = os.path.join(directory_segmentations, image)
        segmentation = get_image_array(file_location)

        total_score = 0
        
        # Separate segmentation in 4 different segmentations.
        seg_0, seg_1, seg_2, seg_3 = separate_segmentation(segmentation)

        # Find the contours in each separate segmentation.
        contours_0_all = find_contours(seg_0, "all")
        contours_1_all = find_contours(seg_1, "all")
        contours_2_all = find_contours(seg_2, "all")
        contours_3_all = find_contours(seg_3, "all")

        num_contours_1 = num_contours(seg_1)
        num_contours_2 = num_contours(seg_2)
        num_contours_3 = num_contours(seg_3)

        num_gaps_1 = check_for_gaps(num_contours_1, contours_1_all, min_gap_size)
        num_gaps_2 = check_for_gaps(num_contours_2, contours_2_all, min_gap_size)
        num_gaps_3 = check_for_gaps(num_contours_3, contours_3_all, min_gap_size)

        num_gaps_12 = check_gap_between_structures(
            seg_1, seg_2, num_gaps_1, num_gaps_2, min_gap_size
        )
        num_gaps_13 = check_gap_between_structures(
            seg_1, seg_3, num_gaps_1, num_gaps_3, min_gap_size
        )
        num_gaps_23 = check_for_gaps_full(
            contours_0_all,
            num_gaps_1,
            num_gaps_2,
            num_gaps_3,
            num_gaps_12,
            num_gaps_13,
            min_gap_size,
        )

        # Check if only one structure of each label is present in segmentation:
        # Check for missing structure.
        total_score += 1 if num_contours_1 == 0 else 0
        total_score += 1 if num_contours_2 == 0 else 0
        total_score += 1 if num_contours_3 == 0 else 0

        # Check for presence of multiple structures.
        total_score += 1 if num_contours_1 > 1 else 0
        total_score += 1 if num_contours_2 > 1 else 0
        total_score += 1 if num_contours_3 > 1 else 0

        # Check for gaps within structures.
        total_score += 1 if num_gaps_1 > 0 else 0
        total_score += 1 if num_gaps_2 > 0 else 0
        total_score += 1 if num_gaps_3 > 0 else 0

        # Check for gap between LV and MYO, LV and LA, and MYO and LA.
        total_score += 1 if num_gaps_12 > 0 else 0
        total_score += 1 if num_gaps_13 > 0 else 0
        total_score += 1 if num_gaps_23 > 0 else 0

        # Extend the overview dictionary with interim results, can be used for analysation. True is good, False is bad.
        overview = [
            num_contours_1 == 0,
            num_contours_2 == 0,
            num_contours_3 == 0,
            num_contours_1 > 1,
            num_contours_2 > 1,
            num_contours_3 > 1,
            num_gaps_1 > 0,
            num_gaps_2 > 0,
            num_gaps_3 > 0,
            num_gaps_12 > 0,
            num_gaps_13 > 0,
            num_gaps_23 > 0,
        ]

        qc_scores.append(total_score)
        overviews[image] = overview

    # Find the frames with a score > 0, these are flagged (for post-processing).
    flagged_frames = sorted([i for i, num in enumerate(qc_scores) if num > 0])

    return qc_scores, overviews, flagged_frames


def get_stats_single_frame_qc(overviews_all):
    """Get statistics of the quality control of segmentation.

    Args:
        overviews_all (dict): Dictionary of all interim results per image.

    Returns:
        stats (dict): Dictionary of statistics of the quality control of segmentation.
    """
    # Count the number of times a certain error occurs.
    cnt_no_LV, cnt_no_MYO, cnt_no_LA = 0, 0, 0
    cnt_multiple_LV, cnt_multiple_MYO, cnt_multiple_LA = 0, 0, 0
    cnt_holes_LV, cnt_holes_MYO, cnt_holes_LA = 0, 0, 0
    cnt_holes_LV_MYO, cnt_holes_LV_LA, cnt_holes_MYO_LA = 0, 0, 0

    for image in overviews_all.keys():
        # Get the overview of one image.
        dict_overall = overviews_all[image]

        for frame in list(dict_overall.keys())[:-4]:
            # Get the overview of one frame.
            res_frame = dict_overall[frame]

            cnt_no_LV += 1 if res_frame[0] == True else 0
            cnt_no_MYO += 1 if res_frame[1] == True else 0
            cnt_no_LA += 1 if res_frame[2] == True else 0

            cnt_multiple_LV += 1 if res_frame[3] == True else 0
            cnt_multiple_MYO += 1 if res_frame[4] == True else 0
            cnt_multiple_LA += 1 if res_frame[5] == True else 0

            cnt_holes_LV += 1 if res_frame[6] == True else 0
            cnt_holes_MYO += 1 if res_frame[7] == True else 0
            cnt_holes_LA += 1 if res_frame[8] == True else 0

            cnt_holes_LV_MYO += 1 if res_frame[9] == True else 0
            cnt_holes_LV_LA += 1 if res_frame[10] == True else 0
            cnt_holes_MYO_LA += 1 if res_frame[11] == True else 0

    # Save the statistics in a dictionary.
    stats = {}
    stats["missing_structure_lv"] = cnt_no_LV
    stats["missing_structure_myo"] = cnt_no_MYO
    stats["missing_structure_la"] = cnt_no_LA
    stats["duplicate_structures_lv"] = cnt_multiple_LV
    stats["duplicate_structures_myo"] = cnt_multiple_MYO
    stats["duplicate_structures_la"] = cnt_multiple_LA
    stats["holes_within_lv"] = cnt_holes_LV
    stats["holes_within_myo"] = cnt_holes_MYO
    stats["holes_within_la"] = cnt_holes_LA
    stats["holes_between_lv_and_myo"] = cnt_holes_LV_MYO
    stats["holes_between_lv_and_la"] = cnt_holes_LV_LA
    stats["holes_between_myo_and_la"] = cnt_holes_MYO_LA

    return stats


def main_single_frame_qc(path_to_segmentations, all_files, views):
    """Main function for single-frame quality control of segmentation.

    Args:
        path_to_segmentations (str): Path to the segmentations.
        all_files (list): List of all files in the directory.
        views (list): List of views of the segmentations.

    Returns:
        single_frame_qc (dict): Dictionary of the results of the single-frame quality control of segmentation.
    """
    single_frame_qc = defaultdict(dict)

    for view in views:
        # Get all files of one view of one person.
        files_of_view = get_list_with_files_of_view(all_files, view)

        # Do single frame QC of segmentation.
        qc_scores, overview, flagged_frames = do_single_frame_qc(
            path_to_segmentations, files_of_view
        )

        # Save the results in a dictionary.
        single_frame_qc["scores"][view] = qc_scores
        single_frame_qc["overview"][view] = overview
        single_frame_qc["flagged_frames"][view] = flagged_frames

    # Get statistics of the quality control of segmentation.
    qc_stats = get_stats_single_frame_qc(single_frame_qc["overview"])
    single_frame_qc["stats"] = qc_stats

    return single_frame_qc
