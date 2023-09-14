
import numpy as np
from scipy.ndimage import map_coordinates



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


