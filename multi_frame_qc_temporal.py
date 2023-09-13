# This script contains functions to flag frames based on their temporal information.
import os
import numpy as np
from collections import defaultdict
from scipy.interpolate import interp1d
from dtaidistance import dtw, dtw_visualisation
from general_utilities import *


def interpolate_missing_areas(timings, areas):
    """Correct for missing values in the area list by interpolating between the non-zero values.

    Args:
        timings (list): List of timings of frames.
        areas (list): List of areas over time for a specific structure.

    Returns:
        areas_interpolated (list): List of areas over time for a specific structure, with missing values interpolated.
    """
    # If there are zero values in the list of areas, interpolate between the non-zero values.
    if min(areas) == 0:
        # Find the indices of the zero values.
        indices_zeros = [index for index, value in enumerate(areas) if value == 0]

        # Find the indices of the non-zero values.
        timings_no_zeros = [
            value for index, value in enumerate(timings) if index not in indices_zeros
        ]
        areas_no_zeros = [
            value for index, value in enumerate(areas) if index not in indices_zeros
        ]

        # Interpolate between the non-zero values.
        f = interp1d(timings_no_zeros, areas_no_zeros, kind="linear")
        areas_interpolated = f(timings)

    else:
        areas_interpolated = areas

    return areas_interpolated


def prepare_area_time_curves(values, frame_times, time_points, nr_time_points=100):
    """Prepare the area-time curves for the DTW analysis.
    
    Args:
        values (list): List of values of the area-time curve of a structure.
        frame_times (list): List of timings of frames.
        time_points (list): List of the first and last frame of the cycle.
        nr_time_points (int): Number of time points to interpolate to.

    Returns:
        values_adjusted (list): List of values of the area-time curve of a structure, prepared for the DTW analysis.    
    """
    # Find the values and timings of the previously selected cycle.
    values_cycle = values[time_points[0]:time_points[1]+1]
    timings_cycle = frame_times[time_points[0]:time_points[1]+1]
    
    # Adjust the timings of the cycle.
    timings_adjusted = [item - timings_cycle[0] for item in timings_cycle]
    total_time = timings_adjusted[-1] - timings_adjusted[0]
    
    # Interpolate between the non-zero values.
    values_interpolated = interpolate_missing_areas(timings_cycle, values_cycle)

    # Create a list of the timings of the cycle.
    timings_for_interpolation = list(np.linspace(0, total_time, nr_time_points, endpoint=True))
    
    # Interpolate between the values of the cycle.
    f = interp1d(timings_adjusted, values_interpolated, kind="linear")

    # Create a list of the values of the cycle.
    values_adjusted = f(timings_for_interpolation)
    
    # Normalise the values of the cycle.
    values_adjusted = normalise_list(values_adjusted)
       
    return values_adjusted


def comp_dtw_distance(cycle_values, atlas):
    """Compute the DTW distance between the area-time curve of a cycle and the atlas.

    Args:
        cycle_values (list): List of values of the area-time curve of a cycle.
        atlas (list): List of values of the area-time curve of the atlas.

    Returns:
        dtw_distance (float): DTW distance between the area-time curve of a cycle and the atlas.
    """
    # Convert to float64 for DTW. 
    atlas_dtw = list(np.array(atlas, dtype="float64"))
    cycle_values_dtw = list(np.array(cycle_values, dtype="float64"))

    dtw_distance = dtw.distance(cycle_values_dtw, atlas_dtw)
    
    # Visualise DTW
    # dtw_path = dtw.warping_path(cycle_values_dtw, atlas_dtw, penalty=0.1) 
    # dtw_visualisation.plot_warping(cycle_values_dtw, atlas_dtw, dtw_path)

    return dtw_distance

def main_multi_frame_qc_temporal(views, cycle_information, segmentation_properties, dicom_properties, atlas_lv, atlas_la):
    """Function to perform the area-time analysis with Dynamic Time Warping (DTW).

    Args:
        views (list): List of views of the segmentations.
        cycle_information (dict): Dictionary containing the information of the cycles.
        segmentation_properties (dict): Dictionary containing the properties of the segmentations.
        dicom_properties (dict): Dictionary containing the properties of the DICOM files.
        atlas_lv (list): List of values of the area-time curve of the atlas for the LV.
        atlas_la (list): List of values of the area-time curve of the atlas for the LA.

    Returns:
        area_time_analysis (dict): Dictionary containing the results of the area-time analysis.
    """
    # Create dictionary to store the area-time analysis.
    area_time_analysis = defaultdict(dict)

    for view in views:
        # Get the properties of the segmentations of the current view.   
        ed_points_cycle = cycle_information["ed_points_selected"][view]
        lv_areas = segmentation_properties["lv_areas"][view]
        la_areas = segmentation_properties["la_areas"][view]
        times_frames = dicom_properties['times_frames'][view]
        
        # Prepare the area-time curves for the DTW analysis.
        lv_areas_cycle_prepared = prepare_area_time_curves(lv_areas, times_frames, ed_points_cycle)
        la_areas_cycle_prepared = prepare_area_time_curves(la_areas, times_frames, ed_points_cycle)
          
        # Compute the DTW distance between the area-time curve of a cycle and the atlas.
        lv_dtw_distance = comp_dtw_distance(lv_areas_cycle_prepared, atlas_lv)
        la_dtw_distance = comp_dtw_distance(la_areas_cycle_prepared, atlas_la)

        # Save the DTW distance in the dictionary.
        area_time_analysis['dtw_lv'][view] = lv_dtw_distance
        area_time_analysis['dtw_la'][view] = la_dtw_distance

    return area_time_analysis
