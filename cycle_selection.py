# Functions to select most appropriate cardiac cycle.
import os
import cv2
import numpy as np
from general_utilities import convert_image, separate_segmentation, find_contours


def calculate_cnr(roi, background):
    """Function to calculate the contrast-to-noise ratio (CNR) of a region of interest (ROI) and background,
    according to the ratio between the mean difference and variance in signal intensities.

    Args:
        roi (numpy array): ROI of the image.
        background (numpy array): Background of the image.

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
        echo_image (numpy array): echo image.
        seg (numpy array): Segmentation of the image.
        desired_labels (list): List of labels to be included in the mask.

    Returns:
        masked_image (numpy array): Masked echo image.
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


def check_for_surrounded_lv(seg_1, seg_2, seg_3, threshold_surrounded_lv=1.0):
    """ Function to check if the left ventricle is fully surrounded by the myocardium and left atrium.

    Args:
        seg_1 (np.ndarray): Segmentation of the image with only label 1, LV.
        seg_2 (np.ndarray): Segmentation of the image with only label 2, MYO.
        seg_3 (np.ndarray): Segmentation of the image with only label 3, LA.
        threshold_surrounded_lv (float): Threshold to adjust robustness of the check (default: 1.0).

    Returns:
        fully_surrounded (bool): Boolean indicating if the left ventricle is fully surrounded by the myocardium and left atrium.   
    """
    # Find the contours of the segmentations. 
    contour_1 = find_contours(seg_1, 'external')
    contour_2 = find_contours(seg_2, 'external')
    contour_3 = find_contours(seg_3, 'external')
    
    # Create lists to store minimum distances between LV and MYO, and MYO and LA. 
    distances_12, distances_13 = [], []
    
    # Check if contours are present for all structures. 
    if len(contour_1) > 0 and len(contour_2) > 0 and len(contour_3) > 0:
        
        for j in range(len(contour_1[0])):
            # Get the coordinates of the left ventricular contour. 
            coordinates_1 = tuple([int(contour_1[0][j][0][0]), int(contour_1[0][j][0][1])])

            # Get the distance between the left ventricular contour and the other contours.
            distance_12 = cv2.pointPolygonTest(contour_2[0], coordinates_1, True)
            distances_12.append(abs(distance_12))

            distance_13 = cv2.pointPolygonTest(contour_3[0], coordinates_1, True)
            distances_13.append(abs(distance_13))
        
        # Check if both structures neighbour the left ventricle. 
        if min(distances_12) <= 1.0 and min(distances_13) <= 1.0:
            # Find the minimum distance between the border of contour 1 and both other contours. 
            distances = [min(pair) for pair in zip(distances_12, distances_13)]
            
            # Check if the maximum distance is smaller than or equal to the threshold. 
            fully_surrounded = max(distances) <= threshold_surrounded_lv
            
        else:
            fully_surrounded = False
            
    else:
        fully_surrounded = False
    
    return fully_surrounded


def check_cut_off_la(seg, contour, rows_to_exclude=5, threshold_cut_off_LA=10):
    """ Function to check if the left atrium is cut off.

    Args:
        seg (np.ndarray): Segmentation of the image.
        contour (list): List of contours of the left atrium.
        rows_to_exclude (int): Number of rows to exclude from the bottom of the segmentation (default: 5).
        threshold_cut_off_LA (int): Threshold to adjust robustness of the check (default: 10).

    Returns:
        no_cut_off_LA (bool): Boolean indicating if the left atrium is cut off.    
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

        # Check if there are less than x bottom rows of the LA present in the segmentation.
        no_cut_off_LA = count < threshold_cut_off_LA
    else:
        no_cut_off_LA = True
    
    return no_cut_off_LA


def find_frames_to_flag(directory_segmentations, images_of_one_person, threshold_surrounded_lv=3.0):      
    # Create lists to store frames that do not meet the QC criteria. 
    flagged_frames_lv, flagged_frames_la = [], []
    
    # Loop over all images of one person
    for frame_nr, image in enumerate(images_of_one_person):
        # Define file location and load segmentation
        file_location = os.path.join(directory_segmentations, image)
        seg = convert_image(file_location)

        # Separate segmentation in 3 different segmentations. 
        _, seg_1, seg_2, seg_3 = separate_segmentation(seg)

        # Find the contours in each separate segmentation
        contours3 = find_contours(seg_3, 'external')      
                                
        # Check if LV is fully surrounded. 
        if not check_LV_fully_surrounded(seg_1, seg_2, seg_3, threshold_surrounded_lv):
            frames_LV_not_surrounded.append(frame_nr)
                 
        # LA cut off check
        if (len(contours3) == 1) and (not check_cut_off_LA(seg3, contours3)):
            frames_LA_cut_off.append(frame_nr)
 
    overview = {}
    overview['Frames NQ LV'] = frames_LV_not_surrounded
    overview['Frames NQ LA'] = frames_LA_cut_off
        
    return overview


def nr_flagged_frames_in_cycle(flagged_frames, first_frame, last_frame):
    """Function to count the number of flagged frames in the cardiac cycle.

    Args:
        flagged_frames (list): List of flagged frames in the image sequence.
        first_frame (int): First frame of the cardiac cycle.
        last_frame (int): Last frame of the cardiac cycle.

    Returns:
        nr_outliers (int): Number of flagged frames in the cardiac cycle.
    """
    nr_outliers = sum([first_frame < number < last_frame for number in flagged_frames])

    return nr_outliers


def score_criteria(my_list, method):
    my_list_rounded = [round(item, 1) for item in my_list]

    min_list = min(my_list_rounded)
    max_list = max(my_list_rounded)

    if method == "max all":
        scores = [
            4 if item == max_list else 0 if item == min_list else 2
            for item in my_list_rounded
        ]
    elif method == "min all":
        scores = [
            4 if item == min_list else 0 if item == max_list else 2
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
    scores_cnr = score_criteria(cnr_cycles, "max all")
    scores_sf_qc = score_criteria(nr_of_outliers_sf_qc, "min all")
    scores_mf_qc = score_criteria(nr_of_outliers_mf_qc, "min all")

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
        count_outliers_lv_mf_qc = nr_outliers_in_cycle(
            frames_to_exclude_lv_mf_qc, ed_points[idx], ed_points[idx + 1]
        )
        count_outliers_la_mf_qc = nr_outliers_in_cycle(
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


def main_cycle_selection(path_to_segmentations, segmentation_properties, QC1, QC2, directory_images, directory_segmentations):
    # Get list of filenames in one folder containing the segmentations.
    all_files = os.listdir(path_to_segmentations)
    patients = sorted(set([i[:29] for i in all_files]))

    cycle_information = {}


    for patient in patients:
        # Get all images of one person.
        images_of_one_person_unsorted = [i for i in all_files if i.startswith(patient)]
        images_of_one_person = sorted(
            images_of_one_person_unsorted, key=lambda x: int(x[30:-7])
        )

    # Get the ED and ES points as well as LV areas for a certain patient
    ED_points = segmentation_properties['ED Points'][patient]
    ES_points = segmentation_properties['ES Points'][patient]
    LV_areas = segmentation_properties['LV areas'][patient]
    
    # Get frames for exclusion QC1
    frames_to_exclude_all_QC1 = QC1['Frames NQ'][patient]
    
    # Get frames for exclusion QC2
    overview_outliers_QC2 = find_outliers_QC2(directory_segmentations, images_of_one_person, threshold_surrounded=3.0)
    frames_to_exclude_LV_QC2 = overview_outliers_QC2['Frames NQ LV']
    frames_to_exclude_LA_QC2 = overview_outliers_QC2['Frames NQ LA']
    
    # Combine frames for exclusion QC1 and QC2
    frames_to_exclude_QC1_QC2 = list(set(frames_to_exclude_all_QC1) | set(frames_to_exclude_LV_QC2) | set(frames_to_exclude_LA_QC2))
    
    # Calculate the CNR for every frame in an image
    CNR_frames = get_cnr_all_frames(directory_images, directory_segmentations, images_of_one_person, frames_to_exclude_QC1_QC2)
    CNR_cycle, ED_selected, ES_selected, score = get_main_cycle(CNR_frames, ED_points, ES_points, LV_areas, frames_to_exclude_all_QC1, frames_to_exclude_LV_QC2, frames_to_exclude_LA_QC2)
    
    CNR_cycles[patient] = CNR_cycle   
    selected_ED[patient] = ED_selected                  
    selected_ES[patient] = ES_selected
    outliers_QC1_all[patient] = frames_to_exclude_all_QC1 
    outliers_QC2_LV[patient] = frames_to_exclude_LV_QC2
    outliers_QC2_LA[patient] = frames_to_exclude_LA_QC2
    scores[patient] = score

    return cycle_information
