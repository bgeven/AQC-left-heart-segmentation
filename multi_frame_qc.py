# This script contains functions to determine the quality control label for each view and for the patient as a whole.
from collections import defaultdict


def count_values_in_range(list_values, start_value, end_value):
    """Count the number of values in a list that fall within a specified range.

    Args:
        list_values (list): The list of values to search through.
        start_value (int): The lower bound of the range.
        end_value (int): The upper bound of the range.

    Returns:
        count (int): The number of values in the list that fall within the specified range.

    Raises:
        ValueError: If start_value is greater than end_value.
    """
    if start_value > end_value:
        raise ValueError("start_value must be less than or equal to end_value")

    count = sum(1 for value in list_values if start_value <= value <= end_value)

    return count


def main_multi_frame_qc(patient, views, cycle_information, multi_frame_qc_structural, multi_frame_qc_temporal, flagged_frame_threshold=2, dtw_thresholds=[1,2]):
    """Main function to assign a quality control label to each view based on structural and temporal analysis.
    The labels are combined to generate an overall patient-level label.

    Args:
        patient (str): The patient ID.
        views (list): The list of views.
        cycle_information (dict): The cycle information.
        multi_frame_qc_structural (dict): The structural quality control information.
        multi_frame_qc_temporal (dict): The temporal quality control information.
        flagged_frame_threshold (int): The threshold for the number of flagged frames within a cycle (default: 2).
        dtw_thresholds (list): The thresholds for the DTW distance between the area-time curve of a cycle and the atlas (default: [1,2]).

    Returns:
        analysis (dict): A dictionary containing analysis results.
            - "label_per_view": A dictionary mapping views to their quality control labels.
            - "label_combined": The combined quality control label for the patient.
    
    """
    analysis = defaultdict(dict)

    for view in views:
        # Load the begin and end points of the cycle.
        ed_points = cycle_information["ed_points_selected"][view]
        
        # Load the flagged frames within the cycle and count the number of flagged frames within the cycle.
        flagged_frames_lv = multi_frame_qc_structural["flagged_frames_lv"][view]
        flagged_frames_la = multi_frame_qc_structural["flagged_frames_la"][view]
        
        nr_flagged_frames_lv_in_cycle = count_values_in_range(flagged_frames_lv, ed_points[0], ed_points[1])
        nr_flagged_frames_la_in_cycle = count_values_in_range(flagged_frames_la, ed_points[0], ed_points[1])

        # Load the DTW distance between the area-time curve of a cycle and the atlas.
        dtw_lv = multi_frame_qc_temporal["dtw_lv"][view]
        dtw_la = multi_frame_qc_temporal["dtw_la"][view]
        
        # Compute the score and label for the current view.
        score_lv = (int(nr_flagged_frames_lv_in_cycle) >= flagged_frame_threshold) + (int(dtw_lv) > dtw_thresholds)
        score_la = (int(nr_flagged_frames_la_in_cycle) >= flagged_frame_threshold) + (int(dtw_la) > dtw_thresholds)

        # Save the score and label for the current view.
        label_lv = score_lv >= 1
        label_la = score_la >= 1
        analysis["label_per_view"][view] = label_lv or label_la
           
    analysis["label_combined"][patient] = analysis["label_per_view"][views[0]] or analysis["label_per_view"][views[1]]

    return analysis
