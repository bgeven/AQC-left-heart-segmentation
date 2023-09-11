# This script containts functions to get properties from a DICOM file.
import os
import pydicom
import numpy as np


def get_frame_times(dicom_data):
    """Function to get the times of each frame in the image sequence.

    Args:
        dicom_data (FileDataset): Pydicom object containing DICOM file data.

    Returns:
        list: Cumulative times of each frame in the image sequence.
    """
    # Retrieve list containing times (in ms) between each frame in the image sequence.
    times_spacing_frames = [float(s) for s in dicom_data.FrameTimeVector._list]

    # Create list with cumulative times of each frame in the image sequence.
    times_frames_added = [
        sum(times_spacing_frames[: i + 1]) for i in range(len(times_spacing_frames))
    ]

    return times_frames_added


def get_pixel_spacing(dicom_data):
    """Function to get the pixel spacing of the image sequence.

    Args:
        dicom_data (Dataset): Pydicom object containing DICOM file data.

    Returns:
        list: Pixel spacing of the image sequence for each dimension.
    """
    pixel_spacing = [float(s) for s in dicom_data.PixelSpacing._list]

    return pixel_spacing


def get_R_wave_frames(dicom_data):
    """Function to get the frame numbers corresponding to the R-wave(s) in the image sequence.

    Args:
        dicom_data (Dataset): Pydicom object containing DICOM file data.

    Returns:
        frames_r_waves (list): Frame numbers corresponding to the R-wave(s) in the image sequence.
    """
    # Create arrays with times of R-wave peaks and times of all frames.
    times_r_waves = np.array(dicom_data.RWaveTimeVector)
    times_all_frames = np.array(get_frame_times(dicom_data))

    # Find the nearest frames corresponding to the timing of the R-wave(s).
    frames_r_waves = [
        int(frame)
        for frame in list(
            np.abs(times_all_frames[:, np.newaxis] - times_r_waves).argmin(axis=0)
        )
    ]

    return frames_r_waves


def main_get_dicom_properties(path_to_dicom_files):
    """Function to get the properties of the DICOM files in a directory.

    The times of each frame, the pixel spacing and the frame numbers corresponding to the R-wave(s) are retrieved.

    Args:
        path_to_dicom_files (str): Path to the directory containing the DICOM files.

    Returns:
        dict: Dictionary containing the properties of the DICOM files.
    """
    # Create dictionary to store the properties of the DICOM files.
    dicom_properties = {}
    dicom_properties["Times Frames"] = {}
    dicom_properties["Pixel Spacing"] = {}
    dicom_properties["Frames R Waves"] = {}

    # Get the DICOM files in the directory.
    dicom_files = os.listdir(path_to_dicom_files)

    for dicom_file in dicom_files:
        dicom_file_location = os.path.join(path_to_dicom_files, dicom_file)

        # Read the DICOM file.
        dicom_data = pydicom.read_file(dicom_file_location, force=True)

        # Get the properties of the DICOM file.
        times_frames = get_frame_times(dicom_data)
        pixel_spacing = get_pixel_spacing(dicom_data)
        frames_r_waves = get_R_wave_frames(dicom_data)

        # Save the properties of the DICOM file in a dictionary.
        dicom_properties["Times Frames"][dicom_file] = times_frames
        dicom_properties["Pixel Spacing"][dicom_file] = pixel_spacing
        dicom_properties["Frames R Waves"][dicom_file] = frames_r_waves

    return dicom_properties
