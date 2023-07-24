# Functions to calculate segmentation parameters from segmentations.
import os
import numpy as np
from scipy.signal import find_peaks
from general_utilities import get_image_array

def get_factor_px2_to_cm2(pixel_spacing):  
    """ Calculate factor to convert pixel size to cm2.	

    Args:
        pixel_spacing (list): Pixel spacing in x and y direction.

    Returns:
        px2cm2_factor (float): Factor to convert pixel size to cm2.
    """
    px2cm2_factor = (pixel_spacing[0] * pixel_spacing[1])/100 # Divide by 100 to convert to cm2.
    
    return px2cm2_factor

def seg_to_area(seg, label, px2cm2_factor):
    """ Calculate the area of a certain label in a segmentation.	

    Args:
        seg (np.array): Segmentation of the image.

    Returns:
        area (float): Area of the label in cm2.
    """
    # Get the pixels of a certain label in a segmentation.
    nr_of_pixels = seg[(seg == label)] 
    
    # Compute the area from the number of pixels and pixel2cm2 factor.
    area = px2cm2_factor * sum(nr_of_pixels) / label 
       
    return area

def get_areas_in_sequence(path_to_seg, frames, label, px2cm2_factor):   
    """ Calculate the area of a certain label in the segmentation of every frame in a sequence.

    Args:
        path_to_seg (str): Directory of the segmentations.	
        frames (list): Frames in the sequence.
        label (int): Label of the segmentation.
        px2cm2_factor (float): Factor to convert pixel size to cm2.

    Returns:
        areas (list): list of areas of the label in cm2.
    """
    areas = [seg_to_area(get_image_array(os.path.join(path_to_seg, frame)), label, px2cm2_factor) for frame in frames]
    
    return areas

def nr_of_ED_points(frames_R_wave, nr_of_frames):
    """ Calculate the number of end-diastolic points based on the number of R-wave peaks.
    
    Args:
        frames_R_wave (list): Frame numbers with R-wave peaks.
        nr_of_frames (int): Number of frames in the sequence.
    
    Returns:
        nr_ED_points (int): number of end-diastolic points.
    """
    nr_of_ED_points = len(frames_R_wave)
   
    # Check if peak towards the end of sequence is present, otherwise add 1 number of ED point. 
    nr_of_ED_points += 1 if abs(frames_R_wave[-1] - nr_of_frames) > 10 else 0
            
    return nr_of_ED_points

def pad_areas(areas):
    """ Pad the list with minimum areas.

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

def find_ES_points(areas, frames_R_wave=[]):  
    """ Find end-systole points, from LV areas.

    Args:
        areas (list): Areas of the label in cm2.
        frames_R_wave (list): Frame numbers with R-wave peaks.
    
    Returns:
        ES_points (list): End-systole points.
    """
    # Find number of ED points and subtract 1 to find number of ES peaks
    if len(frames_R_wave) > 0:
        nr_peaks = nr_of_ED_points(frames_R_wave, len(areas)) - 1
    else:
        nr_peaks = 3 # Set default number of ES points to 3

    # Define the estimated frame difference between the peaks 
    frame_difference = (len(areas) / (nr_peaks+1))
    
    # Find indices of peaks
    peak_indices, _ = find_peaks(-np.array(areas), distance=frame_difference)
    
    # Find the areas that correspond with the indices
    minimal_values = sorted([areas[i] for i in peak_indices], key=float)
    
    # Only take the bottom x (= nr_peaks) minimum values into account and find corresponding frames
    ES_points = sorted([areas.index(j) for j in minimal_values[:nr_peaks]])
    
    return ES_points

def find_ED_points(areas, frames_R_wave=[]):  
    """ Find end-diastole points, from LV areas.

    Args:
        areas (list): Areas of the label in cm2.
        frames_R_wave (list): Frame numbers with R-wave peaks.

    Returns:
        ED_points (list): End-diastole points.
    """
    # Find number of ED points.
    if len(frames_R_wave) > 0:
        nr_peaks = nr_of_ED_points(frames_R_wave, len(areas))
    else:
        nr_peaks = 4 # Set default number of ED points to 4.
        
    # Define the estimated frame difference between the peaks.
    frame_difference = (len(areas)/(nr_peaks))

    # Find indices of peaks, use padding to take side peaks into account.
    areas_padded = areas.copy()
    areas_padded = pad_areas(areas_padded)    
    peak_indices, _ = find_peaks(np.array(areas_padded), distance=frame_difference)
    
    # Find the areas that correspond with the indices, -1 because of padding.
    maximum_values_unsorted = [areas[i-1] for i in peak_indices]
    maximum_values = sorted(maximum_values_unsorted, key=float, reverse=True)
    
    # Only take the top x (= nr_peaks) maximum values into account and find corresponding frames.
    ED_points = sorted([areas.index(j) for j in maximum_values[:nr_peaks]])
       
    return ED_points

def main_get_parameters(path_to_segmentations, dicom_properties):
    """Function to get the segmentation parameters from the segmentations in a directory.

    The areas of the labels in the segmentations are calculated for each frame in the sequence. 
    The ED and ES points are found based on the areas of the LV.

    Args:
        path_to_segmentations (str): Directory containing the segmentations.
        dicom_properties (dict): Dictionary containing the properties of the DICOM files.

    Returns:
        segmentation_properties (dict): Dictionary containing the segmentation parameters.
    """
    # Get list of filenames in one folder containing the segmentations. 
    all_files = os.listdir(path_to_segmentations)
    patients = sorted(set([i[:29] for i in all_files]))

    # Create dictionaries to store the segmentation properties.
    segmentation_properties = {}
    segmentation_properties['LV areas'] = {}
    segmentation_properties['MYO areas'] = {}
    segmentation_properties['LA areas'] = {}
    segmentation_properties['ED Points'] = {}
    segmentation_properties['ES Points'] = {}

    for patient in patients:
        # Get all images of one person.
        images_of_one_person_unsorted = [i for i in all_files if i.startswith(patient)]
        images_of_one_person = sorted(images_of_one_person_unsorted, key=lambda x: int(x[30:-7]))
        
        # Get pixel spacing and ED peaks from ECG R-wave from dicom properties dictionary.
        pixel_spacing = get_factor_px2_to_cm2(dicom_properties['Pixel Spacing'][patient])
        frames_R_waves = dicom_properties['Frames R Waves'][patient]
        
        # Compute the areas per frame for each of the labels.
        LV_areas = get_areas_in_sequence(path_to_segmentations, images_of_one_person, 1, pixel_spacing)
        MYO_areas = get_areas_in_sequence(path_to_segmentations, images_of_one_person, 2, pixel_spacing)
        LA_areas = get_areas_in_sequence(path_to_segmentations, images_of_one_person, 3, pixel_spacing)

        # Find ED and ES points.
        ED_points = find_ED_points(LV_areas, frames_R_waves)
        ES_points = find_ES_points(LV_areas, frames_R_waves)
        
        # Save properties in dictionaries.
        segmentation_properties['LV areas'][patient] = LV_areas
        segmentation_properties['MYO areas'][patient] = MYO_areas
        segmentation_properties['LA areas'][patient] = LA_areas
        segmentation_properties['ED Points'][patient] = ED_points
        segmentation_properties['ES Points'][patient] = ES_points

    return segmentation_properties