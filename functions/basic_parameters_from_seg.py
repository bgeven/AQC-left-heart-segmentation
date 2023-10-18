# This script contains functions to get segmentation parameters from the segmentations.
import os
import numpy as np
from scipy.signal import find_peaks
from collections import defaultdict
from functions.general_utilities import *


def _comp_factor_px_to_cm2(
    pixel_spacing: list[float], conv_factor: int = 100
) -> float:
    """Compute pixel size to cm2 conversion factor.

    Args:
        pixel_spacing (list[float]): Spacing between pixels in x and y direction.
        conv_factor (int): Conversion factor to convert from xxx to cm2 (default: 100).

    Returns:
        px_to_cm2_conv_factor (float): Factor to convert pixel size to cm2.
    """
    px_to_cm2_conv_factor = (
        pixel_spacing[0] * pixel_spacing[1]
    ) / conv_factor  # Convert xxx to cm2.

    return px_to_cm2_conv_factor


def _comp_area_from_seg(seg: np.ndarray, label: int, px_to_cm2_factor: float) -> float:
    """Compute the area of a certain label in a segmentation.

    Args:
        seg (np.ndarray): Segmentation of the echo image.
        label (int): Label of the structure in the segmentation.
        px_to_cm2_factor (float): Factor to convert pixel size to cm2.

    Returns:
        area (float): Area of the structure in cm2.
    """
    # Get the pixels of a certain label in a segmentation.
    nr_of_pixels = seg[(seg == label)]

    # Compute the area from the number of pixels and pixel2cm2 factor.
    area = px_to_cm2_factor * sum(nr_of_pixels) / label

    return area


def _comp_areas_in_sequence(
    path_to_segmentation: str, frames: list[float], label: int, px_to_cm2_factor: float
) -> list[float]:
    """Compute the area of a certain label in the segmentation of every frame in a sequence.

    Args:
        path_to_segmentation (str): Path to the directory containing the segmentations.
        frames (list[float]): Frame numbers present in the sequence.
        label (int): Label of the structure in the segmentation.
        px_to_cm2_factor (float): Factor to convert pixel size to cm2.

    Returns:
        areas (list[float]): Areas of a certain structure for all frames in the sequence, in cm2.
    """
    areas = [
        _comp_area_from_seg(
            convert_image_to_array(os.path.join(path_to_segmentation, frame)),
            label,
            px_to_cm2_factor,
        )
        for frame in frames
    ]

    return areas


def _find_nr_of_ed_points(
    frames_r_wave: list[int], nr_of_frames: int, peak_threshold: int = 10
) -> int:
    """Determine the number of end-diastolic (ED) points based on the number of R-wave peaks.

    Args:
        frames_r_wave (list[int]): Frame numbers with R-wave peaks.
        nr_of_frames (int): Number of frames in the sequence.
        peak_threshold (int): Threshold to account for the last peak if not detected by the find_peaks function (default: 10).

    Returns:
        nr_ed_points (int): Number of ED points.
    """
    nr_of_ed_points = len(frames_r_wave)

    # This is to account for the last peak if not detected by the find_peaks function.
    nr_of_ed_points += (
        1 if abs(frames_r_wave[-1] - nr_of_frames) > peak_threshold else 0
    )

    return nr_of_ed_points


def _pad_areas(areas: list[float]) -> list[float]:
    """Pad the list with minimum areas.

    Args:
        areas (list[float]): Areas of a certain structure for all frames in the sequence, in cm2.

    Returns:
        areas_padded (list[float]): Areas padded with minimum area.
    """
    areas_padded = areas.copy()
    min_value = min(areas)

    # Add the minimum value at the start and end of the list with areas.
    areas_padded.insert(0, min_value)
    areas_padded.append(min_value)

    return areas_padded


def _find_es_points(
    areas: list[float],
    frames_r_wave: list[int] = [],
    nr_peak_type: str = "auto",
    dflt_nr_peaks: int = 1,
) -> list[int]:
    """Determine the end-systole (ES) points from LV areas.

    Args:
        areas (list[float]): Areas of a certain structure for all frames in the sequence, in cm2.
        frames_r_wave (list[int]): Frame numbers with R-wave peaks (default: []).
        nr_peak_type (str): Type of peak determination (default: "auto").
        dflt_nr_peaks (int): Default number of peaks (default: 3).
    
    Returns:
        es_points (list[int]): Frames corresponding to ES phase of the cardiac cycle.
    """
    if nr_peak_type == "auto":
        # Find number of ED points automatically and subtract 1 to find number of ES peaks.
        if len(frames_r_wave) > 0:
            nr_peaks = _find_nr_of_ed_points(frames_r_wave, len(areas)) - 1

        else:
            raise ValueError("nr_peak_type is 'auto', but frames_r_wave is empty.")

    elif nr_peak_type == "force":
        nr_peaks = dflt_nr_peaks  # Set default number of ES points.

    else:
        raise ValueError("nr_peak_type should be 'auto' or 'force'.")

    # Define the estimated frame difference between the peaks.
    frame_difference = len(areas) / (nr_peaks + 1)

    # Find indices of peaks.
    peak_indices, _ = find_peaks(-np.array(areas), distance=frame_difference)

    # Find the areas that correspond with the indices.
    minimal_values = sorted([areas[i] for i in peak_indices], key=float)

    # Only take the bottom x (= nr_peaks) minimum values into account and find corresponding frames.
    es_points = sorted([areas.index(j) for j in minimal_values[:nr_peaks]])

    return es_points


def _find_ed_points(
    areas: list[float],
    frames_r_wave: list[int] = [],
    nr_peak_type: str = "auto",
    dflt_nr_peaks: int = 2,
    
) -> list[int]:
    """Determine the end-diastole (ED) points from LV areas.

    Args:
        areas (list[float]): Areas of a certain structure for all frames in the sequence, in cm2.
        frames_r_wave (list[int]): Frame numbers with R-wave peaks (default: []).
        nr_peak_type (str): Type of peak determination (default: "auto").
        dflt_nr_peaks (int): Default number of ED points (default: 4).

    Returns:
        ed_points (list[int]): Frames corresponding to ED phase of the cardiac cycle.
    """
    # Find number of ED points.
    if nr_peak_type == "auto":
        if len(frames_r_wave) > 0:
            nr_peaks = _find_nr_of_ed_points(frames_r_wave, len(areas))
        else:
            raise ValueError("nr_peak_type is 'auto', but frames_r_wave is empty.")
        
    elif nr_peak_type == "force":
        nr_peaks = dflt_nr_peaks  # Set default number of ED points.

    else:
        raise ValueError("nr_peak_type should be 'auto' or 'force'.")

    # Define the estimated frame difference between the peaks.
    frame_difference = len(areas) / (nr_peaks)

    # Find indices of peaks, use padding to take side peaks into account.
    areas_padded = areas.copy()
    areas_padded = _pad_areas(areas_padded)
    peak_indices, _ = find_peaks(np.array(areas_padded), distance=frame_difference)

    # Find the areas that correspond with the indices, -1 because of padding.
    maximum_values_unsorted = [areas[i - 1] for i in peak_indices]
    maximum_values = sorted(maximum_values_unsorted, key=float, reverse=True)

    # Only take the top x (= nr_peaks) maximum values into account and find corresponding frames.
    ed_points = sorted([areas.index(j) for j in maximum_values[:nr_peaks]])

    return ed_points


def main_derive_parameters(
    path_to_segmentations: str,
    all_files: list[str],
    views: list[str],
    dicom_properties: dict[str, dict[str, list[float]]],
    peak_type_initial: str = "auto",
    dflt_nr_ed_peaks: int = 2,
) -> dict[str, dict[str, list[float]]]:
    """MAIN: Derive basic parameters/properties from the segmentations in a directory.

    The areas of the labels in the segmentations are calculated for each frame in the sequence.
    The end-diastolic (ED) and end-systolic (ES) points are found based on the areas of the left ventricle (LV).

    Args:
        path_to_segmentations (str): Path to the directory containing the segmentations.
        all_files (list[str]): All files in the directory.
        views (list[str]): Plane views of the segmentations.
        dicom_properties (dict[str, dict[str, list[float]]]): Dictionary containing the properties of the DICOM files.
        peak_type_initial (str): Type of peak determination (default: "auto").
        dflt_nr_ed_peaks (int): Default number of ED points (default: 2).

    Returns:
        segmentation_properties (dict[str, dict[str, list[float]]]): Dictionary containing the segmentation parameters.
    """
    # Create dictionaries to store the segmentation properties.
    segmentation_properties = defaultdict(dict)

    for view in views:
        # Get all files of one view of one person.
        files_of_view = get_list_with_files_of_view(all_files, view)

        # Get pixel spacing and ED peaks from ECG R-wave from dicom properties dictionary.
        pixel_spacing = _comp_factor_px_to_cm2(dicom_properties["pixel_spacing"][view])
        frames_r_waves = dicom_properties["frames_r_waves"][view]

        # Compute the areas per frame for each of the labels.
        lv_areas = _comp_areas_in_sequence(
            path_to_segmentations, files_of_view, 1, pixel_spacing
        )
        myo_areas = _comp_areas_in_sequence(
            path_to_segmentations, files_of_view, 2, pixel_spacing
        )
        la_areas = _comp_areas_in_sequence(
            path_to_segmentations, files_of_view, 3, pixel_spacing
        )

        if dicom_properties["frames_r_waves"][view] == [] and peak_type_initial == "auto":
            print("No R-wave frames found, using default number of ED points.")
            peak_type = "force"
        else:
            peak_type = peak_type_initial

        # Find ED and ES points.
        ed_points = _find_ed_points(lv_areas, frames_r_waves, peak_type, dflt_nr_ed_peaks)
        es_points = _find_es_points(lv_areas, frames_r_waves, peak_type, dflt_nr_ed_peaks -1)

        # Save properties in dictionaries.
        segmentation_properties["lv_areas"][view] = lv_areas
        segmentation_properties["myo_areas"][view] = myo_areas
        segmentation_properties["la_areas"][view] = la_areas
        segmentation_properties["ed_points"][view] = ed_points
        segmentation_properties["es_points"][view] = es_points

    return segmentation_properties
