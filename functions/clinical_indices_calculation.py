# This script contains functions to calculate clinical indices of the left ventricle.
import numpy as np
from typing import Union
from collections import defaultdict
from scipy.ndimage import map_coordinates
from functions.general_utilities import *


def _rotate_array(
    array: np.ndarray, angle: float, rotation_point: tuple[float, float]
) -> np.ndarray:
    """Rotate a 2D array by a given angle around a given rotation point.

    Args:
        array (np.ndarray): 2D array to be rotated.
        angle (float): Rotation angle, in radians.
        rotation_point (tuple[float, float]): Rotation point.

    Returns:
        array_rotated (np.ndarray): Rotated array.
    """
    # Create rotation matrix.
    rotation_matrix = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )

    # Create a grid of coordinates for the pixels in the array.
    x_coords, y_coords = np.indices(array.shape)

    # Translate the coordinate system to the center of rotation.
    x_rot, y_rot = rotation_point[1], rotation_point[0]
    x_coords -= x_rot
    y_coords -= y_rot

    # Apply the rotation matrix to the translated coordinates.
    new_coords = np.dot(
        rotation_matrix, np.array([x_coords.flatten(), y_coords.flatten()])
    )

    # Translate the coordinate system back to the original position.
    x_new, y_new = new_coords.reshape(2, array.shape[0], array.shape[1])
    x_new += x_rot
    y_new += y_rot

    # Interpolate the rotated image using the new coordinates.
    array_rotated = map_coordinates(array, [x_new, y_new], order=0)

    return array_rotated


def _find_indices_of_neighbouring_contours(
    x_coordinates_a: np.ndarray,
    y_coordinates_a: np.ndarray,
    x_coordinates_b: np.ndarray,
    y_coordinates_b: np.ndarray,
    distance_threshold: int = 1,
) -> np.ndarray:
    """Find the indices of the coordinates of contour A that are neighboring contour B.

    Args:
        x_coordinates_a (np.ndarray): X-coordinates of contour A.
        y_coordinates_a (np.ndarray): Y-coordinates of contour A.
        x_coordinates_b (np.ndarray): X-coordinates of contour B.
        y_coordinates_b (np.ndarray): Y-coordinates of contour B.
        distance_threshold (float): Threshold distance between two points (default: 1).

    Returns:
        neighbor_indices (np.ndarray): Indices of the coordinates of contour A that are neighboring contour B.
    """
    # Calculate distances from each coordinate of contour A to contour B.
    distances = np.sqrt(
        (x_coordinates_a[:, np.newaxis] - x_coordinates_b[np.newaxis, :]) ** 2
        + (y_coordinates_a[:, np.newaxis] - y_coordinates_b[np.newaxis, :]) ** 2
    )

    # Find the indices where the distance is smaller than a certain threshold.
    neighbor_indices = np.unique(np.where(distances <= distance_threshold)[0])

    return neighbor_indices


def _find_coordinates(contour: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """Find the coordinates of the border of a contour.

    Args:
        contour (list[int]): Contour(s) of a structure.

    Returns:
        x_coordinates (np.ndarray): X-coordinates of the border of the contour.
        y_coordinates (np.ndarray): Y-coordinates of the border of the contour.
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


def _find_mitral_valve_border_coordinates(
    seg_1: np.ndarray,
    seg_2: np.ndarray,
    seg_3: np.ndarray,
    distance_threshold: int = 1,
    max_distance_threshold: int = 10,
    valve_length_threshold: int = 15,
) -> tuple[int, int, int, int]:
    """Find the coordinates of the pixels on the outside of the mitral valve.

    Args:
        seg_1 (np.ndarray): Segmentation of the left ventricle.
        seg_2 (np.ndarray): Segmentation of the myocardium.
        seg_3 (np.ndarray): Segmentation of the left atrium.
        distance_threshold (int): Threshold distance between two points.
        max_distance_threshold (int): Maximum distance between two points.
        valve_length_threshold (int): Minimum length of the mitral valve.

    Returns:
        x_coord_bp_1 (int): X-coordinate of the first border point.
        y_coord_bp_1 (int): Y-coordinate of the first border point.
        x_coord_bp_2 (int): X-coordinate of the second border point.
        y_coord_bp_2 (int): Y-coordinate of the second border point.
    """
    # Find the contours of the structures.
    contours_1 = find_contours(seg_1, spec="external")
    contours_2 = find_contours(seg_2, spec="external")
    contours_3 = find_contours(seg_3, spec="external")

    # Check if all structures are represented by at least one contour.
    if len(contours_1) == 0 or len(contours_2) == 0 or len(contours_3) == 0:
        return 0, 0, 0, 0

    # Find the x and y coordinates of all structures.
    x_coordinates_1, y_coordinates_1 = _find_coordinates(contours_1)
    x_coordinates_2, y_coordinates_2 = _find_coordinates(contours_2)
    x_coordinates_3, y_coordinates_3 = _find_coordinates(contours_3)

    valve_coordinates_not_found = True

    # Continue loop while no common points are found or when threshold distance of LA to LV is larger than maximum threshold distance.
    while valve_coordinates_not_found and distance_threshold < max_distance_threshold:
        neighbor_indices_LV_MYO = _find_indices_of_neighbouring_contours(
            x_coordinates_1,
            y_coordinates_1,
            x_coordinates_2,
            y_coordinates_2,
            distance_threshold,
        )
        neighbor_indices_LV_LA = _find_indices_of_neighbouring_contours(
            x_coordinates_1,
            y_coordinates_1,
            x_coordinates_3,
            y_coordinates_3,
            distance_threshold,
        )

        # Find common points between the neighboring pixels of LV and MYO and LV and LA.
        common_indices = np.intersect1d(neighbor_indices_LV_LA, neighbor_indices_LV_MYO)

        # Check if there are at least 2 common points and the distance between the points is larger than x px,
        # else, increase the threshold distance and continue while loop.
        if (len(common_indices) >= 2) and (
            max(common_indices) - min(common_indices) > valve_length_threshold
        ):
            valve_coordinates_not_found = False
        else:
            distance_threshold += 0.25

    # Check if the number of common points is indeed larger than 2.
    if len(common_indices) >= 2:
        # Find common points and coordinates of the points furthest away from each other.
        common_points = [min(common_indices), max(common_indices)]
        x_coord_bp_1, y_coord_bp_1 = (
            x_coordinates_1[common_points[0]],
            y_coordinates_1[common_points[0]],
        )
        x_coord_bp_2, y_coord_bp_2 = (
            x_coordinates_1[common_points[1]],
            y_coordinates_1[common_points[1]],
        )

    else:
        x_coord_bp_1, y_coord_bp_1, x_coord_bp_2, y_coord_bp_2 = 0, 0, 0, 0

    return x_coord_bp_1, y_coord_bp_1, x_coord_bp_2, y_coord_bp_2


def _find_midpoint_mitral_valve(
    seg_1: np.ndarray, seg_2: np.ndarray, seg_3: np.ndarray
) -> tuple[int, int]:
    """Find the midpoint on the mitral valve.

    Args:
        seg_1 (np.ndarray): Segmentation of the left ventricle.
        seg_2 (np.ndarray): Segmentation of the myocardium.
        seg_3 (np.ndarray): Segmentation of the left atrium.

    Returns:
        coordinates_midpoint (tuple[int, int]): Coordinates of the middle point on the mitral valve.
    """
    # Find the border points on the outside of the mitral valve.
    (
        x_coordinate_mv1,
        y_coordinate_mv1,
        x_coordinate_mv2,
        y_coordinate_mv2,
    ) = _find_mitral_valve_border_coordinates(seg_1, seg_2, seg_3)

    # Find average coordinates of the border points.
    x_avg, y_avg = int((x_coordinate_mv1 + x_coordinate_mv2) / 2), int(
        (y_coordinate_mv1 + y_coordinate_mv2) / 2
    )

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


def _find_apex(
    seg: np.ndarray, x_midpoint: int, y_midpoint: int, mode: str = "before_rotation"
) -> tuple[int, int]:
    """Find the apex of the structure.

    The apex is here defined as the point on the structure that is furthest away from the mitral valve midpoint.

    Args:
        seg (np.ndarray): Segmentation of a certain structure.
        x_midpoint (int): X-coordinate of the mitral valve midpoint.
        y_midpoint (int): Y-coordinate of the mitral valve midpoint.
        mode (str): Mode of the function. Can be either "before_rotation" or "after_rotation" (default: "before_rotation").

    Returns:
        x_apex (int): X-coordinate of the apex.
        y_apex (int): Y-coordinate of the apex.
    """
    # Find contour of specific structure and its x- and y-coordinates.
    contour = find_contours(seg, spec="external")
    x_coords, y_coords = _find_coordinates(contour)

    # Before rotation: apex is not on same vertical line as mitral valve midpoint.
    if mode == "before_rotation":
        # Compute the distance from each coordinate to the mitral valve midpoint and find the index of the point furthest away.
        all_distances = np.sqrt(
            (x_coords - x_midpoint) ** 2 + (y_coords - y_midpoint) ** 2
        )
        idx_max_distance = np.argmax(all_distances)

        # Define the apex coordinates.
        x_apex, y_apex = x_coords[idx_max_distance], y_coords[idx_max_distance]

    # After rotation: apex is on same vertical line as mitral valve midpoint.
    elif mode == "after_rotation":
        # Set x_apex equal to mitral valve midpoint x-coordinate.
        x_apex = x_midpoint

        # Find the y-coordinates of the pixels on the vertical line through the mitral valve midpoint.
        idx = np.where(x_coords == x_apex)
        y_on_line = y_coords[idx]

        # Compute the distance from each point on the line to the mitral valve midpoint and find the index of the point furthest away.
        distances = abs(y_on_line - y_midpoint)
        idx_max_distance = np.argmax(distances)

        # Define the apex coordinates.
        y_apex = y_on_line[idx_max_distance]

    return x_apex, y_apex


def _comp_length_midpoint_apex(
    x_midpoint: int, y_midpoint: int, x_apex: int, y_apex: int, px_to_cm_factor: float
) -> float:
    """Compute the length from the mitral valve midpoint to the apex.

    Args:
        x_midpoint (int): X-coordinate of the mitral valve midpoint.
        y_midpoint (int): Y-coordinate of the mitral valve midpoint.
        x_apex (int): X-coordinate of the apex.
        y_apex (int): Y-coordinate of the apex.
        px_to_cm_factor (float): Conversion factor from pixel to cm.

    Returns:
        length (float): Length from the mitral valve to the apex.
    """
    distance = np.sqrt((x_apex - x_midpoint) ** 2 + (y_apex - y_midpoint) ** 2)
    length = distance * px_to_cm_factor

    return length


def _define_diameters(
    seg: np.ndarray,
    label: int,
    y_midpoint: int,
    y_apex: int,
    px_to_cm_factor: float,
    nr_of_diameters: int = 20,
) -> np.ndarray:
    """Define x diameters perpendicular to the line from the mitral valve to the apex, with equal distances between the diameters.

    Args:
        seg (np.ndarray): Segmentation of a certain structure.
        y_midpoint (int): Y-coordinate of the mitral valve midpoint.
        y_apex (int): Y-coordinate of the apex.
        label (int): Label of the structure in the segmentation.
        px_to_cm_factor (float): Conversion factor from pixel to cm.
        nr_of_diameters (int): Number of diameters to be defined (default: 20).

    Returns:
        diameters (np.ndarray): The diameters perpendicular to the line from the mitral valve to the apex.
    """
    diameters = []

    # Separate the segmentations into different structures.
    _, seg_1, seg_2, seg_3 = separate_segmentation(seg)

    # Find the mitral valve border coordinates.
    x1, _, x2, _ = _find_mitral_valve_border_coordinates(seg_1, seg_2, seg_3)

    # Calculate the initial diameter on the mitral valve.
    diameter_mv = abs(x2 - x1) * px_to_cm_factor
    diameters.append(diameter_mv)

    # Generate points on the line from mitral valve midpoint to apex.
    points_on_L_y = np.linspace(
        y_midpoint, y_apex, nr_of_diameters, endpoint=False, dtype=int
    )

    # Loop over all points on the line from mitral valve to apex.
    for L_y in points_on_L_y[1:]:
        # Create a mask for the specific line and label.
        mask = seg[L_y] == label

        # Calculate the diameter of the structure at the specific line.
        diameter = np.sum(mask) * px_to_cm_factor
        diameters.append(diameter)

    return np.array(diameters)


def _comp_length_and_diameter(
    seg: np.ndarray, pixel_spacing: float, nr_of_diameters: int = 20, label: int = 1
) -> tuple[float, np.ndarray]:
    """Compute the length from mitral valve to apex and the diameters of the segmentation perpendicular to the line from mitral valve to apex.

    Args:
        seg (np.ndarray): Segmentation of the echo image.
        pixel_spacing (float): Spacing between pixels in x and y direction.
        nr_of_diameters (int): Number of diameters to be defined (default: 20).
        label (int): Label of the structure in the segmentation (default: 1 (LV)).

    Returns:
        length (float): Length from the mitral valve to the apex.
        diameters (np.ndarray): Diameters perpendicular to the line from the mitral valve to the apex.
    """
    # Separate the segmentations, each with its own structures.
    _, seg_1, seg_2, seg_3 = separate_segmentation(seg)

    # Find the midpoint of the mitral valve.
    x_midpoint, y_midpoint = _find_midpoint_mitral_valve(seg_1, seg_2, seg_3)

    # Check if the midpoint is not equal to 0 and if there are pixels in the segmentation.
    if x_midpoint == 0 or np.sum(np.array(seg_1) == 1) == 0:
        length = 0
        diameters = np.array([np.nan] * nr_of_diameters)

        return length, diameters

    x_apex, y_apex = _find_apex(seg_1, x_midpoint, y_midpoint, mode="before_rotation")

    # Find angle of rotation and rotate segmentation.
    angle = np.pi + np.arctan2(x_apex - x_midpoint, y_apex - y_midpoint)
    seg_rot = _rotate_array(seg, angle, (x_midpoint, y_midpoint))

    # Find coordinates of apex in rotated segmentation.
    _, seg1_rot, _, _ = separate_segmentation(seg_rot)
    x_apex_rot, y_apex_rot = _find_apex(
        seg1_rot, x_midpoint, y_midpoint, mode="after_rotation"
    )

    # Compute the length from LV apex to middle of mitral valve.
    length = _comp_length_midpoint_apex(
        x_midpoint, y_midpoint, x_apex_rot, y_apex_rot, pixel_spacing
    )

    # Find the diameters perpendicular to the line from mitral valve to apex.
    diameters = _define_diameters(
        seg_rot, label, y_midpoint, y_apex_rot, pixel_spacing, nr_of_diameters
    )

    return length, diameters


def _comp_volume_simpson(
    diameters_a2ch: np.ndarray,
    diameters_a4ch: np.ndarray,
    length_a2ch: float,
    length_a4ch: float,
) -> float:
    """Compute the volume of the structure using the Simpson's method.

    Args:
        diameters_a2ch (np.ndarray): Diameters of the two chamber ED and ES views.
        diameters_a4ch (np.ndarray): Diameters of the four chamber ED and ES views.
        length_a2ch (float): Length from the mitral valve to the apex in the two chamber view.
        length_a4ch (float): Length from the mitral valve to the apex in the four chamber view.

    Returns:
        volume_simpson (float): Volume of the LV calculated using the Simpson's method.
    """
    nr_of_disks = len(diameters_a2ch)

    # Calculate product of the diameter for all disks.
    product_diameters = np.sum(diameters_a2ch * diameters_a4ch)

    # Take the maximum length of the two and four chamber view.
    max_length = max(length_a2ch, length_a4ch)

    # Calculate the volume of the structure using Simpson's method.
    volume_simpson = np.pi * max_length * product_diameters / (4 * nr_of_disks)

    return volume_simpson


def _comp_ejection_fraction(volume_ed: float, volume_es: float) -> float:
    """Compute the ejection fraction of the structure: (EDV-ESV)/EDV

    Args:
        volume_ed (float): End-diastolic volume.
        volume_es (float): End-systolic volume.

    Returns:
        ejection_fraction (float): Ejection fraction.
    """
    ejection_fraction = (volume_ed - volume_es) / volume_ed * 100

    return ejection_fraction


def _process_coordinates(
    coordinates: np.ndarray, neighbor_indices: np.ndarray
) -> np.ndarray:
    """Process the coordinates of the contour by removing the neighboring or overlapping coordinates.

    Args:
        coordinates (np.ndarray): Coordinates of the contour.
        neighbor_indices (np.ndarray): Indices of the neighboring or overlapping coordinates.

    Returns:
        coordinates_continuous (np.ndarray): Coordinates of the first contour without the neighboring or overlapping coordinates.
    """
    # Remove the neighboring or overlapping coordinates from the contour.
    coordinates_removed = np.delete(coordinates, neighbor_indices)

    # Reshuffle the coordinates, so they are all continuous.
    coordinates_continuous = (
        coordinates_removed[neighbor_indices[0] :].tolist()
        + coordinates_removed[: neighbor_indices[0]].tolist()
    )

    return coordinates_continuous


def _comp_circumference(
    seg_a: np.ndarray,
    seg_b: np.ndarray,
    distance_threshold: int = 2,
    neighbor_threshold: int = 10,
) -> float:
    """Compute the circumference of the structure, without including pixels neighboring a specific structure.

    Args:
        seg_a (np.ndarray): Segmentation of the structure.
        seg_b (np.ndarray): Segmentation of the neighboring structure, used to find pixels to exclude.
        distance_threshold (int): Threshold distance between two points (default: 2).
        neighbor_threshold (int): Threshold number of neighboring or overlapping coordinates (default: 10).

    Returns:
        circumference (float): The circumference of the structure.
    """
    # Find the contours of the structures.
    contours_a = find_contours(seg_a, "external")
    contours_b = find_contours(seg_b, "external")

    # Check if both structures exist.
    if not contours_a or not contours_b:
        return np.nan

    # Get the x and y coordinates of the contours
    x_coordinates_a, y_coordinates_a = _find_coordinates(contours_a)
    x_coordinates_b, y_coordinates_b = _find_coordinates(contours_b)

    # Find the indices of neighboring or overlapping coordinates in the first contour.
    neighbor_indices = []
    for i, (x_a, y_a) in enumerate(zip(x_coordinates_a, y_coordinates_a)):
        for x_b, y_b in zip(x_coordinates_b, y_coordinates_b):
            distance = np.sqrt((x_a - x_b) ** 2 + (y_a - y_b) ** 2)
            if distance < distance_threshold:
                neighbor_indices.append(i)
                break

    # Check if there are more than xx neighboring or overlapping coordinates, to ensure that the derived indices are valid.
    if len(neighbor_indices) <= neighbor_threshold:
        return np.nan

    # Process the coordinates of the first contour by removing the neighboring or overlapping coordinates.
    x_coordinates_a_processed = _process_coordinates(x_coordinates_a, neighbor_indices)
    y_coordinates_a_processed = _process_coordinates(y_coordinates_a, neighbor_indices)

    # Calculate the difference between each coordinate.
    dx = np.diff(x_coordinates_a_processed)
    dy = np.diff(y_coordinates_a_processed)

    # Calculate the distance between each coordinate and sum them to get the circumference.
    dist_squared = dx**2 + dy**2
    circumference = np.sum(np.sqrt(dist_squared))

    return circumference


def _comp_global_longitudinal_strain(
    length_circumference_over_time: list[float],
) -> float:
    """Compute the global longitudinal strain (GLS) of the structure for every time frame.

    Args:
        length_circumference_over_time (list): Length of the circumference of the structure for every time frame.

    Returns:
        gls (float): GLS, maximum strain with regards to reference length.

    """
    # Check if the input list is empty or contains only NaN values.
    if not length_circumference_over_time or all(
        np.isnan(length_circumference_over_time)
    ):
        return [np.nan] * len(length_circumference_over_time)

    # Find the first non-NaN distance as the reference distance.
    ref_distance = next(
        (
            distance
            for distance in length_circumference_over_time
            if not np.isnan(distance)
        ),
        np.nan,
    )

    # Calculate gls over time using list comprehension.
    gls_over_time = [
        ((distance - ref_distance) / ref_distance) * 100
        if not np.isnan(distance)
        else np.nan
        for distance in length_circumference_over_time
    ]

    # Find the maximum absolute value of the gls over time.
    max_gls = max([abs(item) for item in gls_over_time])

    return max_gls


def _resize_segmentation(seg: np.ndarray, change_factor: int = 4) -> np.ndarray:
    """Resize the segmentation.

    Args:
        seg (np.ndarray): Segmentation of a certain structure.
        enlarge_factor (int): Factor by which the segmentation is changed (default: 4).

    Returns:
        resized_segmentation (np.ndarray): Resized segmentation.
    """
    resized_segmentation = cv2.resize(
        seg,
        (seg.shape[1] * change_factor, seg.shape[0] * change_factor),
        interpolation=cv2.INTER_NEAREST,
    )

    return resized_segmentation


def _comp_factor_px_to_cm(pixel_spacing: list[float], conv_factor: int = 10) -> float:
    """Compute pixel size to cm conversion factor.

    Args:
        pixel_spacing (list[float]): Spacing between pixels in x and y direction.
        conv_factor (int): Conversion factor to convert from xxx to cm (default: 10).

    Returns:
        pixel_spacing_cm (float): Average pixel spacing in cm.
    """
    pixel_spacing_cm = (pixel_spacing[0] + pixel_spacing[1]) / (2 * conv_factor)

    return pixel_spacing_cm


def _main_cavity_properties(
    path_to_segmentations: str,
    views: list[str],
    all_files: list[str],
    cycle_information: dict[str, dict[str, list[int]]],
    dicom_properties: dict[str, dict[str, list[float]]],
    segmentation_properties: dict[str, dict[str, list[float]]],
    change_factor: int = 4,
    nr_diameters_default: int = 20,
) -> dict[str, dict[str, Union[list[float], float]]]:
    """Determine the diameters and length of the structure for every view.

    Args:
        path_to_segmentations (str): Path to the directory containing the segmentations.
        views (list[str]): Plane views of the segmentations.
        all_files (list[str]): All files in the directory.
        cycle_information (dict[str, dict[str, list[float]]]): Dictionary containing the information of the cardiac cycle.
        dicom_properties (dict[str, dict[str, list[float]]]): Dictionary containing the properties of the DICOM files.
        segmentation_properties (dict[str, dict[str, list[float]]]): Dictionary containing the segmentation parameters.
        change_factor (int): Factor by which the segmentation is changed (default: 4).
        nr_diameters_default (int): Number of diameters to be defined (default: 20).

    Returns:
        cavity_properties (dict[str, dict[str, Union[list[float], float]]): Dictionary containing the diameters, length and area of the cavities for every view.
    """
    cavity_properties = defaultdict(dict)

    for view in views:
        # Initialise lists to store diameters and lengths of both ED frames.
        diameters_ed_both_frames, length_ed_both_frames = [], []

        # Get pixel spacing specific for each image.
        pixel_spacing = (
            _comp_factor_px_to_cm(dicom_properties["pixel_spacing"][view])
            / change_factor
        )

        # Get ED and ES points for each image cycle per view.
        ed_points = cycle_information["ed_points_selected"][view]
        es_point = cycle_information["es_point_selected"][view]

        # Get LA areas for each image cycle per view.
        la_areas = segmentation_properties["la_areas"][view]

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
                    seg = convert_image_to_array(file_location_seg)

                    # Resize segmentation to increase accuracy of length and diameter determination.
                    seg_resized = _resize_segmentation(seg, change_factor)

                    # Determine length and diameters of the structure.
                    length, diameters = _comp_length_and_diameter(
                        seg_resized, pixel_spacing
                    )

                # If the frame is flagged, set length and area to 0 and diameters to NaN values.
                else:
                    length = 0
                    diameters = [np.nan] * nr_diameters_default

                # Store length and diameters in dictionary.
                if idx in ed_points:
                    diameters_ed_both_frames.append([diameters])
                    length_ed_both_frames.append(length)

                elif idx in es_point:
                    cavity_properties["diameters_es"][view] = diameters
                    cavity_properties["length_es"][view] = length

        # Determine the index of the largest length, get the corresponding diameters.
        max_length_idx = np.argmax(length_ed_both_frames)
        cavity_properties["diameters_ed"][view] = diameters_ed_both_frames[
            max_length_idx
        ][0]
        cavity_properties["length_ed"][view] = length_ed_both_frames[max_length_idx]

        # Determine the maximum LA area.
        cavity_properties["max_la_area"][view] = max(
            la_areas[ed_points[0] : ed_points[1] + 1]
        )

    return cavity_properties


def _comp_volume_main(
    views: list[str], cavity_properties: dict[str, dict[str, Union[list[float], float]]]
) -> tuple[float, float]:
    """Compute the volume of the structure using the Simpson's method.

    Args:
        views (list[str]): Plane views of the segmentations.
        cavity_properties (dict[str, dict[str, Union[list[float], float]]): Dictionary containing the diameters, length and area of the cavities for every view.

    Returns:
        volume_simpson_ed (float): End-diastolic volume.
        volume_simpson_es (float): End-systolic volume.
    """
    # Load diameters and lengths of the structure for every view.
    for view in views:
        if "a2ch" in view:
            diameters_ed_a2ch = cavity_properties["diameters_ed"][view]
            diameters_es_a2ch = cavity_properties["diameters_es"][view]
            length_ed_a2ch = cavity_properties["length_ed"][view]
            length_es_a2ch = cavity_properties["length_es"][view]

        elif "a4ch" in view:
            diameters_ed_a4ch = cavity_properties["diameters_ed"][view]
            diameters_es_a4ch = cavity_properties["diameters_es"][view]
            length_ed_a4ch = cavity_properties["length_ed"][view]
            length_es_a4ch = cavity_properties["length_es"][view]

        else:
            raise ValueError("Name of view is not recognised, check name of files.")

    # Check if the diameters and lengths are not equal to 0 and calculate the ED and ES volumes of the structure.
    if length_ed_a2ch != 0 and length_ed_a4ch != 0:
        volume_simpson_ed = _comp_volume_simpson(
            diameters_ed_a2ch, diameters_ed_a4ch, length_ed_a2ch, length_ed_a4ch
        )
    else:
        volume_simpson_ed = np.nan

    if length_es_a2ch != 0 and length_es_a4ch != 0:
        volume_simpson_es = _comp_volume_simpson(
            diameters_es_a2ch, diameters_es_a4ch, length_es_a2ch, length_es_a4ch
        )
    else:
        volume_simpson_es = np.nan

    return volume_simpson_ed, volume_simpson_es


def _comp_circumference_all_frames(
    path_to_segmentations: str, view: str, all_files: list[str]
) -> list[float]:
    """Compute the circumference of the structure for every time frame.

    Args:
        path_to_segmentations (str): Path to the directory containing the segmentations.
        view (str): Plane view of the image/segmentation.
        all_files (list[str]): All files in the directory.

    Returns:
        all_circumferences (list[float]): Circumference of the structure for every time frame.

    TODO: 
        The code to calculate the circumference is not very efficient and slow. This can be improved.
    """
    all_circumferences = []

    # Get all files of one view of one person.
    files_of_view = get_list_with_files_of_view(all_files, view)

    for file in files_of_view:
        # Define file location and load segmentation.
        file_location_seg = os.path.join(path_to_segmentations, file)
        seg = convert_image_to_array(file_location_seg)

        # Separate the segmentations into different structures.
        _, seg_1, _, seg_3 = separate_segmentation(seg)

        # Calculate the circumference of the structure and append it to the list.
        # TODO: Improve efficiency of this circumference code.
        circumference = _comp_circumference(seg_1, seg_3)
        all_circumferences.append(circumference)

    return all_circumferences


def main_computation_clinical_indices(
    path_to_segmentations: str,
    patient: str,
    views: list[str],
    all_files: list[str],
    cycle_information: dict[str, dict[str, list[int]]],
    dicom_properties: dict[str, dict[str, list[float]]],
    segmentation_properties: dict[str, dict[str, list[float]]],
) -> dict[str, dict[str, float]]:
    """MAIN: Compute the clinical indices of the structure.

    This includes the end-diastolic (ED) and end-systolic (ES) volumes, the ejection fraction and the global longitudinal strain.

    Args:
        path_to_segmentations (str): Path to the directory containing the segmentations.
        patient (str): The patient ID.
        views (list[str]): Plane views of the segmentations.
        all_files (list[str]): All files in the directory.
        cycle_information (dict[str, dict[str, list[float]]]): Dictionary containing the information of the cardiac cycle.
        dicom_properties (dict[str, dict[str, list[float]]]): Dictionary containing the properties of the DICOM files.
        segmentation_properties (dict[str, dict[str, list[float]]]): Dictionary containing the segmentation parameters.

    Returns:
        clinical_indices (dict[str, dict[str, float]]): Dictionary containing the clinical indices of the structure.
    """
    clinical_indices = defaultdict(dict)

    # Determine the diameters and length of the structure for every view.
    cavity_properties = _main_cavity_properties(
        path_to_segmentations,
        views,
        all_files,
        cycle_information,
        dicom_properties,
        segmentation_properties,
    )

    # Calculate the volume of the structure and save it in the dictionary.
    volume_simpson_ed, volume_simpson_es = _comp_volume_main(views, cavity_properties)
    clinical_indices[patient]["end_diastolic_volume_[ml]"] = volume_simpson_ed
    clinical_indices[patient]["end_systolic_volume_[ml]"] = volume_simpson_es

    # Calculate the ejection fraction of the structure and save it in the dictionary.
    ejection_fraction = _comp_ejection_fraction(volume_simpson_ed, volume_simpson_es)
    clinical_indices[patient]["ejection_fraction_[%]"] = ejection_fraction

    # Calculate the global longitudinal strain of the structure for both views and save it in the dictionary.
    for view in views:
        circumferences = _comp_circumference_all_frames(
            path_to_segmentations, view, all_files
        )
        global_longitudinal_strain = _comp_global_longitudinal_strain(circumferences)

        if view.endswith("a2ch"):
            clinical_indices[patient][
                "global_long_strain_a2ch_[%]"
            ] = global_longitudinal_strain
            clinical_indices[patient]["maximum_la_area_a2ch_[mm2]"] = cavity_properties[
                "max_la_area"
            ][view]

        elif view.endswith("a4ch"):
            clinical_indices[patient][
                "global_long_strain_a4ch_[%]"
            ] = global_longitudinal_strain
            clinical_indices[patient]["maximum_la_area_a4ch_[mm2]"] = cavity_properties[
                "max_la_area"
            ][view]

    return clinical_indices


def show_clinical_indices(clinical_indices: dict[str, dict[str, float]]) -> None:
    """Show the clinical indices of the patient(s) in a clear way.

    Args:
        clinical_indices (dict[str, dict[str, float]]): Dictionary containing the clinical indices of the structure.
    """
    for patient, values in clinical_indices.items():
        print("{:<30} {:<30}".format("Indices", patient))
        print("-" * 45)
        for index, value in values.items():
            print("{:<30} {:6.1f}".format(index, value))
        print("-" * 45)
