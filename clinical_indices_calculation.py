# This script contains functions to calculate clinical indices of the left ventricle.
import numpy as np
from scipy.ndimage import map_coordinates
from collections import defaultdict
from general_utilities import *


def rotate_array(array, angle, rotation_point):
    """Rotate an array by a given angle around a given rotation point.

    Args:
        seg (np.ndarray): The matrix to be rotated.
        angle (float): The angle in radians.
        rotation_point (tuple): The point around which the matrix is rotated.

    Returns:
        array_rotated (np.ndarray): The rotated matrix.
    """
    # Create rotation matrix. 
    rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    
    # Create a grid of coordinates for the pixels in the array.
    x_coords, y_coords = np.indices(array.shape)
    
    # Translate the coordinate system to the center of rotation.
    rot_x, rot_y = rotation_point[1], rotation_point[0]
    x_coords -= rot_x
    y_coords -= rot_y
    
    # Apply the rotation matrix to the translated coordinates.
    new_coords = np.dot(rotation_matrix, np.array([x_coords.flatten(), y_coords.flatten()]))
    
    # Translate the coordinate system back to the original position.
    new_x, new_y = new_coords.reshape(2, array.shape[0], array.shape[1])
    new_x += rot_x
    new_y += rot_y
    
    # Interpolate the rotated image using the new coordinates.
    array_rotated = map_coordinates(array, [new_x, new_y], order=0)
    
    return array_rotated


def find_indices_of_neighbouring_contours(x_coordinates_a, y_coordinates_a, x_coordinates_b, y_coordinates_b, threshold_distance=1):
    """Find the indices of the coordinates of contour A that are neighboring contour B.

    Args:
        x_coordinates_a (np.ndarray): The x-coordinates of contour A.
        y_coordinates_a (np.ndarray): The y-coordinates of contour A.
        x_coordinates_b (np.ndarray): The x-coordinates of contour B.
        y_coordinates_b (np.ndarray): The y-coordinates of contour B.
        threshold_distance (float): The threshold distance between two points.

    Returns:
        neighbor_indices (np.ndarray): The indices of the coordinates of contour A that are neighboring contour B.
    """
    # Calculate distances from each coordinate of contour A to contour B. 
    distances = np.sqrt((x_coordinates_a[:, np.newaxis] - x_coordinates_b[np.newaxis, :])**2 + (y_coordinates_a[:, np.newaxis] - y_coordinates_b[np.newaxis, :])**2)
    
    # Find the indices where the distance is smaller than a certain threshold. 
    neighbor_indices = np.unique(np.where(distances <= threshold_distance)[0])
    
    return neighbor_indices


def find_coordinates(contour):  
    """Find the coordinates of the border of a contour.

    Args:
        contour (list): The contour of a structure. 
    
    Returns:
        x_coordinates (np.ndarray): The x-coordinates of the border of the contour.
        y_coordinates (np.ndarray): The y-coordinates of the border of the contour.
    """  
    # Check if the structure is represented by a single contour or multiple contours, and find the coordinates accordingly. 
    if len(contour) == 1:
        x_coordinates = contour[0][:, 0, 0]
        y_coordinates = contour[0][:, 0, 1]
        
    elif len(contour) > 1:
        x_coordinates = np.concatenate([c[:, 0, 0] for c in contour])
        y_coordinates = np.concatenate([c[:, 0, 1] for c in contour])

    else:
        x_coordinates, y_coordinates = None, None
        
    return x_coordinates, y_coordinates


def find_mitral_valve_border_coordinates(seg_1, seg_2, seg_3, threshold_distance=1, threshold_length_valve=15):
    """Find the border points on the outside of the mitral valve. 

    Args:
        seg_1 (np.ndarray): The segmentation of the left ventricle.
        seg_2 (np.ndarray): The segmentation of the myocardium.
        seg_3 (np.ndarray): The segmentation of the left atrium.
        threshold_distance (float): The threshold distance between two points.
        threshold_length_valve (float): The minimum length of the mitral valve.

    Returns:
        x1 (int): The x-coordinate of the first border point.
        y1 (int): The y-coordinate of the first border point.
        x2 (int): The x-coordinate of the second border point.
        y2 (int): The y-coordinate of the second border point. 
    """
    # Find the contours of the structures. 
    contours_1 = find_contours(seg_1, spec="external")
    contours_2 = find_contours(seg_2, spec="external")
    contours_3 = find_contours(seg_3, spec="external")
    
    # Check if all structures are represented by at least one contour. 
    if len(contours_1) > 0 and len(contours_2) > 0 and len(contours_3) > 0:
        # Find the x and y coordinates of all structures. 
        x_coordinates_1, y_coordinates_1 = find_coordinates(contours_1)
        x_coordinates_2, y_coordinates_2 = find_coordinates(contours_2)
        x_coordinates_3, y_coordinates_3 = find_coordinates(contours_3)   
    
        valve_coordinates_not_found = True
        
        # Continue loop while no common points are found or when threshold distance of LA to LV is larger than 10 px.
        while valve_coordinates_not_found and threshold_distance < 10:            
            neighbor_indices_LV_MYO = find_indices_of_neighbouring_contours(x_coordinates_1, y_coordinates_1, x_coordinates_2, y_coordinates_2, threshold_distance)
            neighbor_indices_LV_LA = find_indices_of_neighbouring_contours(x_coordinates_1, y_coordinates_1, x_coordinates_3, y_coordinates_3, threshold_distance)
            
            # Find common points between the neighboring pixels of LV and MYO and LV and LA.
            common_indices = np.intersect1d(neighbor_indices_LV_LA, neighbor_indices_LV_MYO)
            
            # Check if there are at least 2 common points and the distance between the points is larger than x px,
            # else, increase the threshold distance and continue while loop.
            if (len(common_indices) >= 2) and (max(common_indices) - min(common_indices) > threshold_length_valve):
                valve_coordinates_not_found = False
            else:
                threshold_distance += 0.25
                
        # Check if the number of common points is indeed larger than 2.
        if len(common_indices) >= 2: 
            # Find common points and coordinates of the points furthest away from each other.
            common_points = [min(common_indices), max(common_indices)]
            x1, y1 = x_coordinates_1[common_points[0]], y_coordinates_1[common_points[0]]
            x2, y2 = x_coordinates_1[common_points[1]], y_coordinates_1[common_points[1]]
            
        else:
            x1, y1, x2, y2 = 0, 0, 0, 0
            
    else:
        x1, y1, x2, y2 = 0, 0, 0, 0

    return x1, y1, x2, y2

 
def find_midpoint_mitral_valve(seg_1, seg_2, seg_3):   
    """Find the middle point on the mitral valve.

    Args:
        seg_1 (np.ndarray): The segmentation of the left ventricle.
        seg_2 (np.ndarray): The segmentation of the myocardium.
        seg_3 (np.ndarray): The segmentation of the left atrium.
    
    Returns:
        coordinates_midpoint (tuple): The coordinates of the middle point on the mitral valve.
    """
    # Find the border points on the outside of the mitral valve.  
    x_coordinate_mv1, y_coordinate_mv1, x_coordinate_mv2, y_coordinate_mv2 = find_mitral_valve_border_coordinates(seg_1, seg_2, seg_3)
            
    # Find average coordinates of the border points. 
    x_avg, y_avg = int((x_coordinate_mv1 + x_coordinate_mv2) / 2), int((y_coordinate_mv1 + y_coordinate_mv2) / 2)
    
    # Find contour of left ventricle. 
    contour_1 = find_contours(seg_1, spec="external")
    contour_array = np.array(contour_1[0])
    
    # Initialise variables for closest point and its distance. 
    coordinates_midpoint = None
    min_distance = float("inf")
    
    # Find the closest point to the average coordinates of the mitral valve.
    for point in contour_array:
        distance = np.sqrt((point[0][0] - x_avg) ** 2 + (point[0][1] - y_avg) ** 2)
        
        if distance < min_distance:
            min_distance = distance
            coordinates_midpoint = tuple(point[0])

    return coordinates_midpoint[0].astype(int), coordinates_midpoint[1].astype(int)


def find_apex(seg, midpoint_x, midpoint_y, mode="before_rotation"):
    """Find the apex of the structure. 

    The apex is here defined as the point on the structure that is furthest away from the mitral valve midpoint.

    Args:
        seg (np.ndarray): The segmentation of the structure.
        midpoint_x (int): The x-coordinate of the mitral valve midpoint.
        midpoint_y (int): The y-coordinate of the mitral valve midpoint.
        mode (str): The mode of the function. Can be either "before_rotation" or "after_rotation".

    Returns:
        x_apex (int): The x-coordinate of the apex.
        y_apex (int): The y-coordinate of the apex.
    """
    # Find contour of specific structure and its x- and y-coordinates. 
    contour = find_contours(seg, spec="external")
    x_coords, y_coords = find_coordinates(contour)
    
    # Before rotation: apex is not on same vertical line as mitral valve midpoint.
    if mode == "before_rotation":
        # Compute the distance from each coordinate to the mitral valve midpoint and find the index of the point furthest away.
        all_distances = np.sqrt((x_coords - midpoint_x)**2 + (y_coords - midpoint_y)**2)
        idx_max_distance = np.argmax(all_distances)
        
        # Define the apex coordinates.
        x_apex, y_apex = x_coords[idx_max_distance], y_coords[idx_max_distance]
      
    # After rotation: apex is on same vertical line as mitral valve midpoint.
    elif mode == "after_rotation": 
        # Set x_apex equal to mitral valve midpoint x-coordinate.
        x_apex = midpoint_x

        # Find the y-coordinates of the pixels on the vertical line through the mitral valve midpoint. 
        idx = np.where(x_coords == x_apex) 
        y_on_line = y_coords[idx]

        # Compute the distance from each point on the line to the mitral valve midpoint and find the index of the point furthest away. 
        distances = abs(y_on_line - midpoint_y)
        idx_max_distance = np.argmax(distances)

        # Define the apex coordinates.
        y_apex = y_on_line[idx_max_distance]

    return x_apex, y_apex


def calculate_length_midpoint_apex(midpoint_x, midpoint_y, x_apex, y_apex, px2cm_factor):
    """Calculate the length from the mitral valve to the apex.

    Args:
        midpoint_x (int): The x-coordinate of the mitral valve midpoint.
        midpoint_y (int): The y-coordinate of the mitral valve midpoint.
        x_apex (int): The x-coordinate of the apex.
        y_apex (int): The y-coordinate of the apex.
        px2cm_factor (float): The pixel spacing.

    Returns:
        length (float): The length from the mitral valve to the apex.
    """
    distance = np.sqrt((x_apex - midpoint_x)**2 + (y_apex - midpoint_y)**2)
    length = distance * px2cm_factor

    return length


def define_diameters(seg, label, midpoint_y, y_apex, px2cm_factor, nr_of_diameters=20):
    """Define x diameters perpendicular to the line from the mitral valve to the apex.
    
    Args:
        seg (np.ndarray): The segmentation of the structure.
        midpoint_y (int): The y-coordinate of the mitral valve midpoint.
        y_apex (int): The y-coordinate of the apex.
        label (int): The label of the structure.
        px2cm_factor (float): The pixel spacing.
        nr_of_diameters (int): The number of diameters to be defined (default: 20).

    Returns:
        diameters (list): The diameters perpendicular to the line from the mitral valve to the apex.        
    """
    diameters = [] 
 
    # Separate the segmentations into different structures.  
    _, seg_1, seg_2, seg_3 = separate_segmentation(seg)
    
    # Find the mitral valve border coordinates. 
    x1, _, x2, _ = find_mitral_valve_border_coordinates(seg_1, seg_2, seg_3)
    
    # Calculate the initial diameter on the mitral valve. 
    diameter_mv = abs(x2 - x1) * px2cm_factor
    diameters.append(diameter_mv)
    
    # Generate points on the line from mitral valve midpoint to apex. 
    points_on_L_y = np.linspace(midpoint_y, y_apex, nr_of_diameters, endpoint=False, dtype = int)
 
    # Loop over all points on the line from mitral valve to apex. 
    for L_y in points_on_L_y[1:]:
        # Create a mask for the specific line and label. 
        mask = (seg[L_y] == label)

        # Calculate the diameter of the structure at the specific line. 
        diameter = np.sum(mask) * px2cm_factor                
        diameters.append(diameter)
                    
    return np.array(diameters)


def determine_length_and_diameter(seg, pixel_spacing, nr_of_diameters=20, label=1):
    """Calculate the length from mitral valve to apex and the 20 diameters of the segmentation perpendicular to the line from mitral valve to apex.

    Args:
        seg (np.ndarray): The segmentation of the structure.
        pixel_spacing (float): The pixel spacing.
        nr_of_diameters (int): The number of diameters to be defined (default: 20).
        label (int): The label of the structure (default: 1 (LV)).
    
    Returns:    
        length (float): The length from the mitral valve to the apex.
        diameters (list): The diameters perpendicular to the line from the mitral valve to the apex.
    """
    # Separate the segmentations, each with its own structures. 
    _, seg_1, seg_2, seg_3 = separate_segmentation(seg)
    
    # Find the midpoint of the mitral valve. 
    midpoint_x, midpoint_y = find_midpoint_mitral_valve(seg_1, seg_2, seg_3)

    # Check if the midpoint is not equal to 0 and if there are pixels in the segmentation. 
    if midpoint_x != 0 and np.sum(np.array(seg_1) == 1) > 0:
        x_apex, y_apex = find_apex(seg_1, midpoint_x, midpoint_y, mode="before_rotation")
                
        # Find angle of rotation and rotate segmentation. 
        angle = np.pi + np.arctan2(x_apex - midpoint_x, y_apex - midpoint_y)
        seg_rot = rotate_array(seg, angle, (midpoint_x, midpoint_y))
            
        # Find coordinates of apex in rotated segmentation. 
        _, seg1_rot, _, _ = separate_segmentation(seg_rot)
        x_apex_rot, y_apex_rot = find_apex(seg1_rot, midpoint_x, midpoint_y, mode="after_rotation")
    
        # Compute the length from LV apex to middle of mitral valve. 
        length = calculate_length_midpoint_apex(midpoint_x, midpoint_y, x_apex_rot, y_apex_rot, pixel_spacing)

        # Find the diameters perpendicular to the line from mitral valve to apex.
        diameters = define_diameters(seg_rot, label, midpoint_y, y_apex_rot, pixel_spacing, nr_of_diameters)

    # If the midpoint is equal to 0 or there are no pixels in the segmentation, return NaN values.
    else:
        length = 0
        diameters = [np.nan] * nr_of_diameters
        
    return length, diameters


def comp_volume_simpson(diameters_a2ch, diameters_a4ch, length_a2ch, length_a4ch):
    """Calculate the volume of the structure using the Simpson's method.

    Args:
        diameters_a2ch (np.ndarray): The diameters of the two chamber ED and ES views.
        diameters_a4ch (np.ndarray): The diameters of the four chamber ED and ES views.
        length_a2ch (float): The length from the mitral valve to the apex in the two chamber view.
        length_a4ch (float): The length from the mitral valve to the apex in the four chamber view.

    Returns:
        volume_simpson (float): The volume of the LV calculated using the Simpson's method.
    """
    nr_of_disks = len(diameters_a2ch)

    # Calculate product of the diameter for all disks. 
    product_diameters = np.sum(diameters_a2ch * diameters_a4ch)

    # Take the maximum length of the two and four chamber view.
    max_length = max(length_a2ch, length_a4ch)

    # Calculate the volume of the structure using Simpson's method.
    volume_simpson = np.pi * max_length * product_diameters / (4 * nr_of_disks) 
    
    return volume_simpson


def comp_ejection_fraction(volume_ED, volume_ES):
    """Calculate the ejection fraction of the structure: (EDV-ESV)/EDV

    Args:
        volume_ED (float): The end-diastolic volume.
        volume_ES (float): The end-systolic volume.

    Returns:
        ejection_fraction (float): The ejection fraction.
    """
    ejection_fraction = (volume_ED - volume_ES)/volume_ED * 100
    
    return ejection_fraction


def process_coordinates(coordinates, neighbor_indices):
    """Process the coordinates of the contour by removing the neighboring or overlapping coordinates.

    Args:
        coordinates (np.ndarray): The coordinates of the contour.
        neighbor_indices (np.ndarray): The indices of the neighboring or overlapping coordinates.

    Returns:
        coordinates_continuous (np.ndarray): The coordinates of the first contour without the neighboring or overlapping coordinates.
    """
    # Remove the neighboring or overlapping coordinates from the contour. 
    coordinates_removed = np.delete(coordinates, neighbor_indices)
    
    # Reshuffle the coordinates, so they are all continuous. 
    coordinates_continuous = coordinates_removed[neighbor_indices[0]:].tolist() + coordinates_removed[:neighbor_indices[0]].tolist()

    return coordinates_continuous


def comp_circumference(seg_A, seg_B, threshold_distance=1.5):
    """Calculate the circumference of the structure, without including pixels neighboring a specific structure.

    Args:
        seg_A (np.ndarray): The segmentation of the structure.
        seg_B (np.ndarray): The segmentation of the neighboring structure, used to find pixels to exclude. 
        threshold_distance (float): The threshold distance between two points (default: 1.5).

    Returns:
        circumference (float): The circumference of the structure.
    """
    # Find the contours of the structures.
    contours_A = find_contours(seg_A, "external")
    contours_B = find_contours(seg_B, "external")
    
    # Check if both structures exist. 
    if not contours_A or not contours_B:
        return np.nan

    # Get the x and y coordinates of the contours
    x_coordinates_A, y_coordinates_A = find_coordinates(contours_A)
    x_coordinates_B, y_coordinates_B = find_coordinates(contours_B)
    
    # Find the indices of neighboring or overlapping coordinates in the first contour. 
    neighbor_indices = []
    for i, (x_A, y_A) in enumerate(zip(x_coordinates_A, y_coordinates_A)):
        for x_B, y_B in zip(x_coordinates_B, y_coordinates_B):
            distance = np.sqrt((x_A - x_B) ** 2 + (y_A - y_B) ** 2)
            if distance < threshold_distance:
                neighbor_indices.append(i)
                break
    
    # Check if there are more than 10 neighboring or overlapping coordinates.
    if len(neighbor_indices) > 10:
        # Process the coordinates of the first contour by removing the neighboring or overlapping coordinates.
        x_coordinates_A_processed = process_coordinates(x_coordinates_A, neighbor_indices)
        y_coordinates_A_processed = process_coordinates(y_coordinates_A, neighbor_indices)

        # Calculate the difference between each coordinate.
        dx = np.diff(x_coordinates_A_processed)
        dy = np.diff(y_coordinates_A_processed)
        
        # Calculate the distance between each coordinate and sum them to get the circumference.  
        dist_squared = dx**2 + dy**2
        circumference = np.sum(np.sqrt(dist_squared))

    else: 
        circumference = np.nan

    return circumference


def comp_global_longitudinal_strain(length_circumference_over_time):
    """Calculate the global longitudinal strain (GLS) of the structure for every time frame.

    Args:
        length_circumference_over_time (list): The length of the circumference of the structure for every time frame.

    Returns:
        gls (float): The global longitudinal strain, maximum strain with regards to reference length.

    """
    # Check if the input list is empty or contains only NaN values.
    if not length_circumference_over_time or all(np.isnan(length_circumference_over_time)):
        return [np.nan] * len(length_circumference_over_time)
    
    # Find the first non-NaN distance as the reference distance.
    ref_distance = next((distance for distance in length_circumference_over_time if not np.isnan(distance)), np.nan)

    # Calculate gls over time using list comprehension.
    gls_over_time = [((distance - ref_distance) / ref_distance) * 100 if not np.isnan(distance) else np.nan for distance in length_circumference_over_time]

    # Find the maximum absolute value of the gls over time.
    max_gls = max([abs(item) for item in gls_over_time])
    
    return max_gls


def resize_segmentation(seg, enlarge_factor=4):
    """Resize the segmentation.

    Args:
        seg (np.ndarray): The segmentation of the structure.
        enlarge_factor (int): The factor by which the segmentation is enlarged (default: 4).
    
    Returns:
        resized_segmentation (np.ndarray): The resized segmentation.
    """
    resized_segmentation = cv2.resize(seg, (seg.shape[1] * enlarge_factor, seg.shape[0] * enlarge_factor), interpolation=cv2.INTER_NEAREST)

    return resized_segmentation


def main_diameter_length_determination(path_to_segmentations, views, all_files, cycle_information, dicom_properties, enlarge_factor=4):
    """Determine the diameters and length of the structure for every view.

    Args:
        path_to_segmentations (str): The path to the folder containing the segmentations.
        views (list): The views of the structure.
        all_files (list): The list of all files in the folder.
        cycle_information (dict): The dictionary containing the information of the cardiac cycle.
        dicom_properties (dict): The dictionary containing the DICOM properties.
        enlarge_factor (int): The factor by which the segmentation is enlarged (default: 4).

    Returns:
        diameters_and_lengths (dict): The dictionary containing the diameters and length of the structure for every view.
    """
    diameters_and_lengths = defaultdict(dict)

    for view in views:
        # Initialise lists to store diameters and lengths of both ED frames. 
        diameters_ed_both_frames, length_ed_both_frames = [], []
        
        # Get pixel spacing specific for each image. 
        pixel_spacing = conv_pixel_spacing_to_cm(dicom_properties["pixel_spacing"][view]) / enlarge_factor
        
        # Get ED and ES points for each image cycle per view. 
        ed_points = cycle_information["ed_points_selected"][view]
        es_point = cycle_information["es_point_selected"][view]
         
        # Get frames to exclude from analysis. 
        frames_to_exclude = cycle_information["flagged_frames_combined"][view]
                
        # Get list of all files of one view of one person.
        files_of_view = get_list_with_files_of_view(all_files, view)
        
        for idx, filename in enumerate(files_of_view): 
            # Check if the frame is an ED or ES frame and if it is not flagged.
            if idx in ed_points or idx in es_point:
                if idx not in frames_to_exclude:
                    # Get segmentation of specific frame.
                    file_location_seg = os.path.join(path_to_segmentations, filename)
                    seg = get_image_array(file_location_seg)

                    # Resize segmentation to increase accuracy of length and diameter determination.
                    seg_resized = resize_segmentation(seg, enlarge_factor)

                    # Determine length and diameters of the structure.
                    length, diameters = determine_length_and_diameter(seg_resized, pixel_spacing)

                # If the frame is flagged, set length to 0 and diameters to NaN values.
                else:
                    length, diameters = 0, [np.nan] * 20

                # Store length and diameters in dictionary.
                if idx in ed_points:
                    diameters_ed_both_frames.append([diameters])
                    length_ed_both_frames.append(length)

                elif idx in es_point:
                    diameters_and_lengths["diameters_es"][view] = diameters
                    diameters_and_lengths["length_es"][view] = length

        # Determine the index of the largest length and get the corresponding diameters.
        max_length_idx = np.argmax(length_ed_both_frames)
        diameters_and_lengths["diameters_ed"][view] = diameters_ed_both_frames[max_length_idx][0]
        diameters_and_lengths["length_ed"][view] = length_ed_both_frames[max_length_idx]                  

    return diameters_and_lengths


def main_volume_calculation(views, diameters_and_lengths):
    """Calculate the volume of the structure using the Simpson's method.

    Args:
        views (list): The views of the structure.
        diameters_and_lengths (dict): The dictionary containing the diameters and length of the structure for every view.

    Returns:
        volume_simpson_ed (float): The end-diastolic volume.
        volume_simpson_es (float): The end-systolic volume.
    """
    volume_simpson_ed = np.nan
    volume_simpson_es = np.nan

    # Load diameters and lengths of the structure for every view.
    for view in views:
        if "a2ch" in view:
            diameters_ed_a2ch = diameters_and_lengths["diameters_ed"][view]
            diameters_es_a2ch = diameters_and_lengths["diameters_es"][view]
            length_ed_a2ch = diameters_and_lengths["length_ed"][view]
            length_es_a2ch = diameters_and_lengths["length_es"][view]

        elif "a4ch" in view:
            diameters_ed_a4ch = diameters_and_lengths["diameters_ed"][view]
            diameters_es_a4ch = diameters_and_lengths["diameters_es"][view]
            length_ed_a4ch = diameters_and_lengths["length_ed"][view]
            length_es_a4ch = diameters_and_lengths["length_es"][view]

        else:
            raise ValueError("Name of view is not recognised, check this.")

    # Check if the diameters and lengths are not equal to 0 and calculate the ED and ES volumes of the structure.
    if length_ed_a2ch != 0 and length_ed_a4ch != 0:
        volume_simpson_ed = comp_volume_simpson(diameters_ed_a2ch, diameters_ed_a4ch, length_ed_a2ch, length_ed_a4ch)

    if length_es_a2ch != 0 and length_es_a4ch != 0:
        volume_simpson_es = comp_volume_simpson(diameters_es_a2ch, diameters_es_a4ch, length_es_a2ch, length_es_a4ch)

    return volume_simpson_ed, volume_simpson_es


def main_circumference_all_frames(path_to_segmentations, view, all_files):
    """Calculate the circumference of the structure for every time frame.

    Args:
        path_to_segmentations (str): The path to the folder containing the segmentations.
        view (str): The view of the structure.
        all_files (list): The list of all files in the folder.

    Returns:
        all_circumferences (list): The circumference of the structure for every time frame.
    """
    all_circumferences = []
    
    # Get all files of one view of one person.
    files_of_view = get_list_with_files_of_view(all_files, view)

    for file in files_of_view:
        # Define file location and load segmentation. 
        file_location_seg = os.path.join(path_to_segmentations, file)
        seg = get_image_array(file_location_seg)
        
        # Separate the segmentations into different structures.
        _, seg_1, _, seg_3 = separate_segmentation(seg)
        
        # Calculate the circumference of the structure and append it to the list.
        circumference = comp_circumference(seg_1, seg_3)   
        all_circumferences.append(circumference)

    return all_circumferences


def main_computation_clinical_indices(path_to_segmentations, patient, views, all_files, diameters_and_lengths):
    """Calculate the clinical indices of the structure.

    This includes the ED and ES volumes, the ejection fraction and the global longitudinal strain.

    Args:
        path_to_segmentations (str): The path to the folder containing the segmentations.
        patient (str): The patient ID.
        views (list): The views of the structure.
        all_files (list): The list of all files in the folder.
        diameters_and_lengths (dict): The dictionary containing the diameters and length of the structure for every view.

    Returns:
        clinical_indices (dict): The dictionary containing the clinical indices of the structure.
    """
    clinical_indices = defaultdict(dict)

    # Calculate the volume of the structure and save it in the dictionary.
    volume_simpson_ed, volume_simpson_es = main_volume_calculation(views, diameters_and_lengths)
    clinical_indices["volume_ed"][patient] = volume_simpson_ed
    clinical_indices["volume_es"][patient] = volume_simpson_es
        
    # Calculate the ejection fraction of the structure and save it in the dictionary.
    ejection_fraction = comp_ejection_fraction(volume_simpson_ed, volume_simpson_es)
    clinical_indices["ejection_fraction"][patient] = ejection_fraction
    
    # Calculate the global longitudinal strain of the structure for both views and save it in the dictionary.
    for view in views:
        circumferences = main_circumference_all_frames(path_to_segmentations, view, all_files)
        global_longitudinal_strain = comp_global_longitudinal_strain(circumferences)
    
        if view.endswith("a2ch"):
            clinical_indices["gls_a2ch"][patient] = global_longitudinal_strain

        elif view.endswith("a4ch"):
            clinical_indices["gls_a4ch"][patient] = global_longitudinal_strain

    return clinical_indices
