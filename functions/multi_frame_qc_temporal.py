# This script contains functions to flag frames based on their temporal information.
import numpy as np
from collections import defaultdict
from scipy.interpolate import interp1d
from dtaidistance import dtw, dtw_visualisation
from functions.general_utilities import *


def _interpolate_missing_areas(timings: list[float], areas: list[float]) -> list[float]:
    """Interpolate between the non-zero values in the area list to correct for missing values.

    Args:
        timings (list[float]): Timings of frames.
        areas (list[float]): Areas over time for a specific structure.

    Returns:
        areas_interpolated (list[float]): Areas over time for a specific structure, with missing values interpolated.
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


def _prepare_area_time_curves(
    values: list[float],
    frame_times: list[float],
    time_points: list[int],
    nr_time_points: int = 100,
) -> list[float]:
    """Prepare the area-time curves for the DTW analysis.

    Args:
        values (list[float]): Values of the area-time curve of a structure.
        frame_times (list[float]): Timings of frames.
        time_points (list[int]): First and last frame of the cycle.
        nr_time_points (int): Number of time points to interpolate to.

    Returns:
        values_adjusted (list[float]): Values of the area-time curve of a structure, prepared for the DTW analysis.
    """
    if len(frame_times) == 0:
        frame_times = list(range(len(values)))

    # Find the values and timings of the previously selected cycle.
    values_cycle = values[time_points[0] : time_points[1] + 1]
    timings_cycle = frame_times[time_points[0] : time_points[1] + 1]

    # Adjust the timings of the cycle.
    timings_adjusted = [item - timings_cycle[0] for item in timings_cycle]
    total_time = timings_adjusted[-1] - timings_adjusted[0]

    # Interpolate between the non-zero values.
    values_interpolated = _interpolate_missing_areas(timings_cycle, values_cycle)

    # Create a list of the timings of the cycle.
    timings_for_interpolation = list(
        np.linspace(0, total_time, nr_time_points, endpoint=True)
    )

    # Interpolate between the values of the cycle.
    f = interp1d(timings_adjusted, values_interpolated, kind="linear")

    # Create a list of the values of the cycle.
    values_adjusted = f(timings_for_interpolation)

    # Normalise the values of the cycle.
    values_adjusted = normalise_list(values_adjusted)

    return values_adjusted


def _comp_dtw_distance(cycle_values: list[float], atlas: list[float]) -> float:
    """Compute the dynamic time warping (DTW) distance between the area-time curve of a cycle and the atlas.

    Args:
        cycle_values (list[float]): Values of the area-time curve of a cycle.
        atlas (list[float]): Values of the area-time curve of the atlas.

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


def main_multi_frame_qc_temporal(
    views: list[str],
    cycle_information: dict[str, dict[str, list[int]]],
    segmentation_properties: dict[str, dict[str, list[int]]],
    dicom_properties: dict[str, dict[str, list[int]]],
    atlas_lv: list[float],
    atlas_la: list[float],
) -> dict[str, dict[str, float]]:
    """MAIN: Do multi-frame QC assessment based on temporal criteria (dynamic time warping (DTW) distance).
    
    Args:
        views (list[str]): Plane views of the segmentations.
        cycle_information (dict[str, dict[str, list[float]]]): Dictionary containing the information of the cardiac cycle.
        segmentation_properties (dict[str, dict[str, list[float]]]): Dictionary containing the segmentation parameters.
        dicom_properties (dict[str, dict[str, list[float]]]): Dictionary containing the properties of the DICOM files.
        atlas_lv (list[float]): Values of the area-time curve of the atlas for the LV.
        atlas_la (list[float]): Values of the area-time curve of the atlas for the LA.

    Returns:
        area_time_analysis (dict[str, dict[str, list[float]]]): Dictionary containing the results of the area-time analysis.
    """
    # Create dictionary to store the area-time analysis.
    area_time_analysis = defaultdict(dict)

    for view in views:
        # Get the properties of the segmentations of the current view.
        ed_points_cycle = cycle_information["ed_points_selected"][view]
        lv_areas = segmentation_properties["lv_areas"][view]
        la_areas = segmentation_properties["la_areas"][view]
        times_frames = dicom_properties["times_frames"][view]

        # Prepare the area-time curves for the DTW analysis.
        lv_areas_cycle_prepared = _prepare_area_time_curves(
            lv_areas, times_frames, ed_points_cycle
        )
        la_areas_cycle_prepared = _prepare_area_time_curves(
            la_areas, times_frames, ed_points_cycle
        )

        # Compute the DTW distance between the area-time curve of a cycle and the atlas.
        lv_dtw_distance = _comp_dtw_distance(lv_areas_cycle_prepared, atlas_lv)
        la_dtw_distance = _comp_dtw_distance(la_areas_cycle_prepared, atlas_la)

        # Save the DTW distance in the dictionary.
        area_time_analysis["dtw_lv"][view] = lv_dtw_distance
        area_time_analysis["dtw_la"][view] = la_dtw_distance

    return area_time_analysis
