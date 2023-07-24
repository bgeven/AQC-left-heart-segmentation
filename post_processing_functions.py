# Functions to do post-processing
import os
import cv2
import numpy as np
from general_utilities import find_contours, find_largest_contour, get_image_array, separate_segmentation

def find_centroid(seg):   
    """ Find centroid of structures present in segmentation.

    Args:
        seg (numpy array): Segmentation of structures.
    
    Returns:
        centroid_x (int): x-coordinate of centroid.
        centroid_y (int): y-coordinate of centroid.
    """
    # Find the contours of the structures in segmentation         
    contours = find_contours(seg)
    
    # If no contour present, assign middle coordinate value nan 
    if len(contours) == 0:
        centroid_x, centroid_y = np.nan, np.nan
    
    # If one or multiple contours are present, continue:
    else:
        # Find largest contour in segmentation
        main_contour = find_largest_contour(contours)
        
        # Find moment and middle coordinates of largest contour
        moments_main_contour = cv2.moments(main_contour)
        centroid_x = int(moments_main_contour['m10'] / moments_main_contour['m00'])
        centroid_y = int(moments_main_contour['m01'] / moments_main_contour['m00'])
        
    return centroid_x, centroid_y

def get_mean_centroids(path_to_segmentations, images_of_one_person, frames_to_process):  
    # Create dictionary to store middle points of structures
    centroids = {1: ([], []), 2: ([], []), 3: ([], [])}
    
    for frame_nr, image in enumerate(images_of_one_person):
        # Define file location and load segmentation. 
        file_location_seg = os.path.join(path_to_segmentations, image)
        seg = get_image_array(file_location_seg)

        # Separate the segmentations, each with its own structures
        seg0, seg1, seg2, seg3 = separate_segmentation(seg)

        # Prevent from not selecting a centroid at all when all frames are detected as erroneous. 
        if len(frames_to_process) < round(0.95*len(images_of_one_person)):
            if frame_nr not in frames_to_process:               
                # Get middle coordinate of each separate segmentation. 
                for structure in centroids:
                    centroid_x, centroid_y = find_centroid(locals()[f"seg{structure}"])
                    centroids[structure][0].append(centroid_x)
                    centroids[structure][1].append(centroid_y)
        else:
            # Get middle coordinate of each separate segmentation. 
            for structure in centroids:
                centroid_x, centroid_y = find_centroid(locals()[f"seg{structure}"])
                centroids[structure][0].append(centroid_x)
                centroids[structure][1].append(centroid_y)
        
    # Compute average middle point of each structure. 
    mean_centroids = {structure: (int(np.nanmean(coords[0])), int(np.nanmean(coords[1]))) for structure, coords in centroids.items()}

    return mean_centroids