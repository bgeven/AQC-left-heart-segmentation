import os
import cv2
import numpy as np
from general_utilities import (
    find_contours,
    combine_segmentations,
    get_image_array,
    separate_segmentation,
)


def num_contours(seg):
    """Find the number of external contours in a segmentation.

    Args:
        seg (numpy array): Segmentation of the image.

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


def check_gap_between_structures(segA, segB, num_gapsA, num_gapsB, min_size=[1, 1]):
    """Check if gaps are present between two structures.

    Args:
        segA (numpy array): Segmentation of the first structure.
        segB (numpy array): Segmentation of the second structure.
        num_gapsA (int): Number of gaps in the first structure.
        num_gapsB (int): Number of gaps in the second structure.
        min_size (list): Minimum size of a gap.

    Returns:
        num_gaps (int): Number of gaps between the two structures.
    """
    # Create total segmentation of different combinations of structures
    total_seg = combine_segmentations([segA, segB], "no difference")

    # Find contours within combined segmentations 1 and 2
    contoursAB = find_contours(total_seg, "all")
    filtered_contours = [
        contour
        for contour in contoursAB
        if cv2.contourArea(contour) >= (min_size[0] * min_size[1])
    ]

    # Find number of gaps within segmentations A and B
    nr_of_gaps = num_gapsA + num_gapsB

    # Define the presence of a gap between 2 structures as the number of contours seg A and B
    # minus number of gaps within seg A and B minus 1 (because of minimal number of contours A and B of 1).
    num_gaps = len(filtered_contours) - nr_of_gaps - 1

    return num_gaps


def check_for_gaps_full(
    contours, num_gaps1, num_gaps2, num_gaps3, num_gaps12, num_gaps13, min_size=[1, 1]
):
    """Check if gaps are present in a segmentation.

    Args:
        contours (list): List of all contours in the segmentation.
        num_gaps1 (int): Number of gaps in the first structure.
        num_gaps2 (int): Number of gaps in the second structure.
        num_gaps3 (int): Number of gaps in the third structure.
        num_gaps12 (int): Number of gaps between the first and second structure.
        num_gaps13 (int): Number of gaps between the first and third structure.
        min_size (list): Minimum size of a gap.

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
        - num_gaps1
        - num_gaps2
        - num_gaps3
        - num_gaps12
        - num_gaps13
        - 1
    )

    return num_other_gaps


def do_single_frame_QC(directory_segmentations, images_of_one_person):
    """Do single-frame quality control of segmentation.

    This quality control is based on the following criteria:
    1. Only one structure of each label is present in segmentation.
    2. No gaps within and between structures.

    Args:
        directory_segmentations (str): Directory of the segmentations.
        images_of_one_person (list): List of all images of one person.
        min_gap_size (list): Minimum size of a gap.

    Returns:
        QC_scores (list): List of QC scores per image.
        overviews (dict): Dictionary of all interim results per image.
        flagged_frames (list): List of frames with a QC score > 0.
    """
    QC_scores, overviews = [], {}

    # Loop over all images of one person
    for image in images_of_one_person:
        # Define file location and load segmentation.
        file_location = os.path.join(directory_segmentations, image)
        segmentation = get_image_array(file_location)

        total_score = 0
        min_gap_size = [2, 2]

        # Separate segmentation in 4 different segmentations.
        seg0, seg1, seg2, seg3 = separate_segmentation(segmentation)

        # Find the contours in each separate segmentation.
        contours0_all = find_contours(seg0, "all")
        contours1_all = find_contours(seg1, "all")
        contours2_all = find_contours(seg2, "all")
        contours3_all = find_contours(seg3, "all")

        num_contours1 = num_contours(seg1)
        num_contours2 = num_contours(seg2)
        num_contours3 = num_contours(seg3)

        num_gaps1 = check_for_gaps(num_contours1, contours1_all, min_gap_size)
        num_gaps2 = check_for_gaps(num_contours2, contours2_all, min_gap_size)
        num_gaps3 = check_for_gaps(num_contours3, contours3_all, min_gap_size)

        num_gaps12 = check_gap_between_structures(
            seg1, seg2, num_gaps1, num_gaps2, min_gap_size
        )
        num_gaps13 = check_gap_between_structures(
            seg1, seg3, num_gaps1, num_gaps3, min_gap_size
        )
        num_gaps23 = check_for_gaps_full(
            contours0_all,
            num_gaps1,
            num_gaps2,
            num_gaps3,
            num_gaps12,
            num_gaps13,
            min_gap_size,
        )

        # Check if only one structure of each label is present in segmentation:
        # Check for missing structure.
        total_score += 1 if num_contours1 == 0 else 0
        total_score += 1 if num_contours2 == 0 else 0
        total_score += 1 if num_contours3 == 0 else 0

        # Check for presence of multiple structures.
        total_score += 1 if num_contours1 > 1 else 0
        total_score += 1 if num_contours2 > 1 else 0
        total_score += 1 if num_contours3 > 1 else 0

        # Check for gaps within structures.
        total_score += 1 if num_gaps1 > 0 else 0
        total_score += 1 if num_gaps2 > 0 else 0
        total_score += 1 if num_gaps3 > 0 else 0

        # Check for gap between LV and MYO, LV and LA, and MYO and LA.
        total_score += 1 if num_gaps12 > 0 else 0
        total_score += 1 if num_gaps13 > 0 else 0
        total_score += 1 if num_gaps23 > 0 else 0

        # Extend the overview dictionary with interim results, can be used for analysation. True is good, False is bad.
        overview = [
            num_contours1 == 0,
            num_contours2 == 0,
            num_contours3 == 0,
            num_contours1 > 1,
            num_contours2 > 1,
            num_contours3 > 1,
            num_gaps1 > 0,
            num_gaps2 > 0,
            num_gaps3 > 0,
            num_gaps12 > 0,
            num_gaps13 > 0,
            num_gaps23 > 0,
        ]

        QC_scores.append(total_score)
        overviews[image] = overview

    # Find the frames with a score > 0, these are flagged (for post-processing).
    flagged_frames = sorted([i for i, num in enumerate(QC_scores) if num > 0])

    return QC_scores, overviews, flagged_frames


def get_stats_QC1(overviews_all):
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
    stats["Missing structure LV"] = cnt_no_LV
    stats["Missing structure MYO"] = cnt_no_MYO
    stats["Missing structure LA"] = cnt_no_LA
    stats["Duplicate structures LV"] = cnt_multiple_LV
    stats["Duplicate structures MYO"] = cnt_multiple_MYO
    stats["Duplicate structures LA"] = cnt_multiple_LA
    stats["Holes within LV"] = cnt_holes_LV
    stats["Holes within MYO"] = cnt_holes_MYO
    stats["Holes within LA"] = cnt_holes_LA
    stats["Holes between LV and MYO"] = cnt_holes_LV_MYO
    stats["Holes between LV and LA"] = cnt_holes_LV_LA
    stats["Holes between MYO and LA"] = cnt_holes_MYO_LA

    return stats


def main_single_frame_QC(path_to_segmentations):
    """Main function for single-frame quality control of segmentation.

    Args:
        path_to_segmentations (str): Path to the segmentations.

    Returns:
        single_frame_QC (dict): Dictionary of the results of the single-frame quality control of segmentation.
    """
    single_frame_QC = {}
    single_frame_QC["QC_Scores"] = {}
    single_frame_QC["Overviews"] = {}
    single_frame_QC["Flagged_Frames"] = {}

    # Get list of filenames in one folder containing the segmentations.
    all_files = os.listdir(path_to_segmentations)
    patients = sorted(set([i[:29] for i in all_files]))

    for patient in patients:
        # Get the images per patient and sort these based on frame number.
        images_of_one_person_unsorted = [i for i in all_files if i.startswith(patient)]
        images_of_one_person = sorted(
            images_of_one_person_unsorted, key=lambda x: int(x[30:-7])
        )

        # Do single frame QC of segmentation.
        QC_scores, overview, flagged_frames = do_single_frame_QC(
            path_to_segmentations, images_of_one_person
        )

        # Save the results in a dictionary.
        single_frame_QC["QC_Scores"][patient] = QC_scores
        single_frame_QC["Overviews"][patient] = overview
        single_frame_QC["Flagged_Frames"][patient] = flagged_frames

    # Get statistics of the quality control of segmentation.
    QC_stats = get_stats_QC1(single_frame_QC["Overviews"])
    single_frame_QC["QC_Stats"] = QC_stats

    return single_frame_QC
