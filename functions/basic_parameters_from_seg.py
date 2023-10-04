# This script contains functions to get segmentation parameters from the segmentations.
import os
import numpy as np
from scipy.signal import find_peaks
from collections import defaultdict
from functions.general_utilities import get_image_array, get_list_with_files_of_view


def comp_factor_px2_to_cm2(pixel_spacing):
    """Calculate factor to convert pixel size to cm2.

    Args:
        pixel_spacing (list): Pixel spacing in x and y direction.

    Returns:
        px2cm2_factor (float): Factor to convert pixel size to cm2.
    """
    px2cm2_factor = (
        pixel_spacing[0] * pixel_spacing[1]
    ) / 100  # Divide by 100 to convert to cm2.

    return px2cm2_factor


def comp_area_from_seg(seg, label, px2cm2_factor):
    """Calculate the area of a certain label in a segmentation.

    Args:
        seg (np.ndarray): Segmentation of the echo image.

    Returns:
        area (float): Area of the label in cm2.
    """
    # Get the pixels of a certain label in a segmentation.
    nr_of_pixels = seg[(seg == label)]

    # Compute the area from the number of pixels and pixel2cm2 factor.
    area = px2cm2_factor * sum(nr_of_pixels) / label

    return area


def comp_areas_in_sequence(path_to_segmentation, frames, label, px2cm2_factor):
    """Calculate the area of a certain label in the segmentation of every frame in a sequence.

    Args:
        path_to_segmentation (str): Path to the segmentations.
        frames (list): Frames in the sequence.
        label (int): Label of the segmentation.
        px2cm2_factor (float): Factor to convert pixel size to cm2.

    Returns:
        areas (list): list of areas of the label in cm2.
    """
    areas = [
        comp_area_from_seg(
            get_image_array(os.path.join(path_to_segmentation, frame)),
            label,
            px2cm2_factor,
        )
        for frame in frames
    ]

    return areas


def find_nr_of_ed_points(frames_r_wave, nr_of_frames, threshold_peak=10):
    """Determine the number of end-diastolic points based on the number of R-wave peaks.

    Args:
        frames_r_wave (list): Frame numbers with R-wave peaks.
        nr_of_frames (int): Number of frames in the sequence.
        threshold_peak (int): Threshold to account for the last peak if not detected by the find_peaks function (default: 10).

    Returns:
        nr_ed_points (int): number of end-diastolic points.
    """
    nr_of_ed_points = len(frames_r_wave)

    # This is to account for the last peak if not detected by the find_peaks function.
    nr_of_ed_points += (
        1 if abs(frames_r_wave[-1] - nr_of_frames) > threshold_peak else 0
    )

    return nr_of_ed_points


def pad_areas(areas):
    """Pad the list with minimum areas.

    Args:
        areas (list): Areas of the label in cm2.

    Returns:
        areas_padded (list): Areas of the label in cm2, padded with minimum area.
    """
    areas_padded = areas.copy()
    min_value = min(areas)

    # Add the minimum value at the start and end of the list with areas.
    areas_padded.insert(0, min_value)
    areas_padded.append(min_value)

    return areas_padded


def find_es_points(areas, frames_r_wave=[]):
    """Determine the end-systole points from LV areas.

    Args:
        areas (list): Areas of the label in cm2.
        frames_r_wave (list): Frame numbers with R-wave peaks.

    Returns:
        es_points (list): End-systole (ES) points.
    """
    # Find number of ED points and subtract 1 to find number of ES peaks.
    if len(frames_r_wave) > 0:
        nr_peaks = find_nr_of_ed_points(frames_r_wave, len(areas)) - 1
    else:
        nr_peaks = 3  # Set default number of ES points to 3

    # Define the estimated frame difference between the peaks.
    frame_difference = len(areas) / (nr_peaks + 1)

    # Find indices of peaks.
    peak_indices, _ = find_peaks(-np.array(areas), distance=frame_difference)

    # Find the areas that correspond with the indices.
    minimal_values = sorted([areas[i] for i in peak_indices], key=float)

    # Only take the bottom x (= nr_peaks) minimum values into account and find corresponding frames.
    es_points = sorted([areas.index(j) for j in minimal_values[:nr_peaks]])

    return es_points


def find_ed_points(areas, frames_r_wave=[]):
    """Determine the end-diastole (ED) points from LV areas.

    Args:
        areas (list): Areas of the label in cm2.
        frames_r_wave (list): Frame numbers with R-wave peaks.

    Returns:
        ed_points (list): ED points.
    """
    # Find number of ED points.
    if len(frames_r_wave) > 0:
        nr_peaks = find_nr_of_ed_points(frames_r_wave, len(areas))
    else:
        nr_peaks = 4  # Set default number of ED points to 4.

    # Define the estimated frame difference between the peaks.
    frame_difference = len(areas) / (nr_peaks)

    # Find indices of peaks, use padding to take side peaks into account.
    areas_padded = areas.copy()
    areas_padded = pad_areas(areas_padded)
    peak_indices, _ = find_peaks(np.array(areas_padded), distance=frame_difference)

    # Find the areas that correspond with the indices, -1 because of padding.
    maximum_values_unsorted = [areas[i - 1] for i in peak_indices]
    maximum_values = sorted(maximum_values_unsorted, key=float, reverse=True)

    # Only take the top x (= nr_peaks) maximum values into account and find corresponding frames.
    ed_points = sorted([areas.index(j) for j in maximum_values[:nr_peaks]])

    return ed_points


def main_get_parameters(path_to_segmentations, all_files, views, dicom_properties):
    """Main function to get the segmentation parameters from the segmentations in a directory.

    The areas of the labels in the segmentations are calculated for each frame in the sequence.
    The ED and ES points are found based on the areas of the LV.

    Args:
        path_to_segmentations (str): Directory containing the segmentations.
        all_files (list): List of all files in the directory.
        views (list): List of views of the segmentations.
        dicom_properties (dict): Dictionary containing the properties of the DICOM files.

    Returns:
        segmentation_properties (dict): Dictionary containing the segmentation parameters.
    """
    # Create dictionaries to store the segmentation properties.
    segmentation_properties = defaultdict(dict)

    for view in views:
        # Get all files of one view of one person.
        files_of_view = get_list_with_files_of_view(all_files, view)

        # Get pixel spacing and ED peaks from ECG R-wave from dicom properties dictionary.
        pixel_spacing = comp_factor_px2_to_cm2(dicom_properties["pixel_spacing"][view])
        frames_r_waves = dicom_properties["frames_r_waves"][view]

        # Compute the areas per frame for each of the labels.
        lv_areas = comp_areas_in_sequence(
            path_to_segmentations, files_of_view, 1, pixel_spacing
        )
        myo_areas = comp_areas_in_sequence(
            path_to_segmentations, files_of_view, 2, pixel_spacing
        )
        la_areas = comp_areas_in_sequence(
            path_to_segmentations, files_of_view, 3, pixel_spacing
        )

        # Find ED and ES points.
        ed_points = find_ed_points(lv_areas, frames_r_waves)
        es_points = find_es_points(lv_areas, frames_r_waves)

        # Save properties in dictionaries.
        segmentation_properties["lv_areas"][view] = lv_areas
        segmentation_properties["myo_areas"][view] = myo_areas
        segmentation_properties["la_areas"][view] = la_areas
        segmentation_properties["ed_points"][view] = ed_points
        segmentation_properties["es_points"][view] = es_points

    return segmentation_properties
