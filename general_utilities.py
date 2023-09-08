import cv2
import numpy as np
import SimpleITK as sitk

def get_image_array(file_location):
    """ Convert nifti or dicom file to 2D array.

    Args:
        file_location (str): Location of the nifti or dicom file.

    Returns:
        image (np.ndarray): 2D array of the image.
    """
    # Load image and convert to array. 
    itk_image = sitk.ReadImage(file_location) 
    image_array = sitk.GetArrayFromImage(itk_image)
    
    # If array has a pseudo 3D shape, remove extra dimension (output nnU-Net v1).
    if len(image_array.shape) > 2:
        image = np.reshape(image_array.transpose((1, 2, 0)), (image_array.shape[1], image_array.shape[2]))
    else:
        image = image_array
        
    return image

def separate_segmentation(seg):
    """ Separate the LV, MYO and LA from the segmentation into separate segmentations.

    Args:
        seg (np.ndarray): Segmentation of the image.

    Returns:
        seg_0 (np.ndarray): Segmentation of the image with only label 0 (background).
        seg_1 (np.ndarray): Segmentation of the image with only label 1 (LV).
        seg_2 (np.ndarray): Segmentation of the image with only label 2 (MYO).
        seg_3 (np.ndarray): Segmentation of the image with only label 3 (LA).
    """
    seg_0 = np.where(seg == 0, seg, 1).astype(np.uint8)
    seg_1 = np.where(seg == 1, seg, 0)
    seg_2 = np.where(seg == 2, seg, 0)
    seg_3 = np.where(seg == 3, seg, 0)
    
    return seg_0, seg_1, seg_2, seg_3

def find_contours(seg, spec='all'):
    """ Find the contours within the segmentation.

    Args:
        seg (np.ndarray): Segmentation of the image.
        spec (str): Specification of the contours to find.
    
    Returns:
        contours (list): List of contours.
    """
    # Define the retrieval modes. 
    retrieval_modes = {
        'all': cv2.RETR_LIST,
        'external': cv2.RETR_EXTERNAL,
        'tree': cv2.RETR_TREE}
    
    # Find contours based on the specified mode. 
    contours, _ = cv2.findContours(seg, retrieval_modes.get(spec, cv2.RETR_LIST), cv2.CHAIN_APPROX_NONE)
    
    return contours

def find_largest_contour(contours):
    """ Find the largest contour within a list of contours, based on area.

    Args:
        contours (list): List of contours.
    
    Returns:
        largest_contour (list): Largest contour.
    """	
    largest_contour = max(contours, key=cv2.contourArea)
    
    return largest_contour

def combine_segmentations(segmentations, typeOfCombination='difference', labels=[1,2,3]):
    """ Combine the segmentations of the LV, MYO and LA into one segmentation.	

    A distinction can be made between the following types of combination:
    - difference: each structure gets other value.
    - difference with overlap: each structure gets other value, but overlap is accounted for.
    - no difference: all structures get same value.

    Args:
        segmentations (list): List of segmentations of the image.
        typeOfCombination (str): Type of combination of the segmentations.
        labels (list): List of labels of the segmentations.
    
    Returns:
        total_seg (np.ndarray): Segmentation of the image with all structures.
    """	
    # Give the first structure the value of the first label.
    if np.amax(segmentations[0]) == labels[0]:
        total_seg = segmentations[0].copy()
    else:
        total_seg = segmentations[0].copy() * labels[0]
        
    # Each structure gets other value.
    if typeOfCombination == 'difference':
        for nr, seg in enumerate(segmentations[1:]):
            if seg.max() == labels[nr+1]:
                total_seg += seg
            else:
                total_seg += seg * labels[nr+1]

    # Each structure gets other value, but overlap is accounted for. 
    if typeOfCombination == 'difference with overlap':
        for nr, seg in enumerate(segmentations[1:]):
            if seg.max() == labels[nr+1]:
                total_seg += seg
            else:
                total_seg += seg * labels[nr+1]
                
            if labels[nr+1] == 2:
                total_seg[total_seg > labels[nr+1]] = labels[nr]
            elif labels[nr+1] == 3:
                total_seg[total_seg > labels[nr+1]] = labels[nr+1]
    
    # All structures get same value.
    elif typeOfCombination == 'no difference':
        for seg in segmentations[1:]:
            total_seg += seg

        total_seg[total_seg > 0] = 1
        
    return total_seg

def find_coordinates_of_holes(seg): 
    """ Find the coordinates of the holes in a segmentation.

    Args:
        seg (np.ndarray): Segmentation of the image.

    Returns:
        coordinates_holes (tuple): Coordinates of the holes in the segmentation.
    """
    # Set all values larger than 0 to 1. 
    seg_same_val = seg.copy()
    seg_same_val[seg_same_val > 0] = 1
    
    # Find the contours of the structures in full segmentation. 
    contours = find_contours(seg_same_val, 'external')

    coordinates_holes_x_all, coordinates_holes_y_all = np.array([]), np.array([])
        
    for contour in contours:       
        # Create a mask from the contour. 
        mask = cv2.drawContours(np.zeros_like(seg_same_val), [contour], 0, 255, -1)
        
        # Find the positions of all the zero pixels within the contour. 
        coordinates_holes_contour = np.where((mask == 255) & (seg_same_val == 0))

        coordinates_holes_x, coordinates_holes_y = coordinates_holes_contour[0], coordinates_holes_contour[1]
        
        coordinates_holes_x_all = np.append(coordinates_holes_x_all, coordinates_holes_x) 
        coordinates_holes_y_all = np.append(coordinates_holes_y_all, coordinates_holes_y) 
    
    coordinates_holes = (coordinates_holes_x_all.astype('int64'), coordinates_holes_y_all.astype('int64'))
    
    return coordinates_holes