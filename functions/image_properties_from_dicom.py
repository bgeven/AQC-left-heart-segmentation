# This script containts functions to get properties from a DICOM file.
import os
import pydicom
import numpy as np
from collections import defaultdict


def _get_frame_times(dicom_data: pydicom.FileDataset) -> list[float]:
    """Get the times of each frame in the image sequence.

    Args:
        dicom_data (pydicom.FileDataset): Pydicom object containing DICOM file data.

    Returns:
        times_frames_added (list[float]): Cumulative times of each frame in the image sequence.
    """
    # Retrieve list containing times (in ms) between each frame in the image sequence.
    times_spacing_frames = [float(s) for s in dicom_data.FrameTimeVector._list]

    # Create list with cumulative times of each frame in the image sequence.
    times_frames_added = [
        sum(times_spacing_frames[: i + 1]) for i in range(len(times_spacing_frames))
    ]

    return times_frames_added


def _get_pixel_spacing(dicom_data: pydicom.FileDataset) -> list[float]:
    """Get the pixel spacing of the image sequence.

    Args:
        dicom_data (pydicom.FileDataset): Pydicom object containing DICOM file data.

    Returns:
        pixel_spacing (list[float]): Pixel spacing of the image sequence for each dimension.
    """
    pixel_spacing = [float(s) for s in dicom_data.PixelSpacing._list]

    return pixel_spacing


def _get_R_wave_frames(dicom_data: pydicom.FileDataset) -> list[int]:
    """Get the frame numbers corresponding to the R-wave(s) in the image sequence.

    Args:
        dicom_data (pydicom.FileDataset): Pydicom object containing DICOM file data.

    Returns:
        frames_r_waves (list[int]): Frame numbers corresponding to the R-wave(s) in the image sequence.
    """
    # Create arrays with times of R-wave peaks and times of all frames.
    times_r_waves = np.array(dicom_data.RWaveTimeVector)
    times_all_frames = np.array(_get_frame_times(dicom_data))

    # Find the nearest frames corresponding to the timing of the R-wave(s).
    frames_r_waves = [
        int(frame)
        for frame in list(
            np.abs(times_all_frames[:, np.newaxis] - times_r_waves).argmin(axis=0)
        )
    ]

    return frames_r_waves


def main_get_dicom_properties(
    path_to_dicom_files: str,
    views: list[str],
    default_pixel_spacing: list[float] = [0.1, 0.1],
    default_frames_r_waves: list[int] = [],
) -> dict[str, dict[str, list[float]]]:
    """MAIN: Get the properties of the DICOM files in a directory.

    The times of each frame, the pixel spacing and the frame numbers corresponding to the R-wave(s) are retrieved.

    Args:
        path_to_dicom_files (str): Path to the directory containing the DICOM files.
        views (list[str]): List of views of the segmentations.
        default_pixel_spacing (list[float]): Default pixel spacing of the image sequence for each dimension (default: [0.1, 0.1]).
        default_frames_r_waves (list[int]): Default frame numbers corresponding to the R-wave(s) in the image sequence (default: []).

    Returns:
        dicom_properties (dict[str, dict[str, list[int]]]): Dictionary containing the properties of the DICOM files.
    """
    # Create dictionary to store the properties of the DICOM files.
    dicom_properties = defaultdict(dict)

    # Get the DICOM files in the directory.
    dicom_files = os.listdir(path_to_dicom_files)

    # Check if the DICOM files are in the directory.
    if len(dicom_files) > 0:
        for dicom_file in dicom_files:
            dicom_file_location = os.path.join(path_to_dicom_files, dicom_file)

            # Read the DICOM file.
            dicom_data = pydicom.read_file(dicom_file_location, force=True)

            # Get the properties of the DICOM file.
            times_frames = _get_frame_times(dicom_data)
            pixel_spacing = _get_pixel_spacing(dicom_data)
            frames_r_waves = _get_R_wave_frames(dicom_data)

            # Save the properties of the DICOM file in a dictionary.
            dicom_properties["times_frames"][dicom_file] = times_frames
            dicom_properties["pixel_spacing"][dicom_file] = pixel_spacing
            dicom_properties["frames_r_waves"][dicom_file] = frames_r_waves

    # If no DICOM files are in the directory, use default values.
    else:
        for view in views:
            dicom_properties["pixel_spacing"][view] = default_pixel_spacing
            dicom_properties["frames_r_waves"][view] = default_frames_r_waves
            dicom_properties["times_frames"][view] = []

            print("No DICOM files found in the directory, so default values are used.")
            print("WARNING: The pixel spacing is set to {} and {}, in x- and y-direction respectively.".format(default_pixel_spacing[0], default_pixel_spacing[1]))
            print("WARNING: Please check default number of cardiac cycles (dflt_nr_ed_peaks).")
            print("WARNING: If these default values are incorrect, the calculated clinical indices will be incorrect.")

    return dicom_properties
