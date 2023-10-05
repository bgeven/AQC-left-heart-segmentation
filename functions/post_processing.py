# This script contains functions to do post-processing on the segmentations.
import os
import cv2
import shutil
import numpy as np
import SimpleITK as sitk
from functions.general_utilities import *


def find_centroid(seg: np.ndarray) -> tuple[int, int]:
    """Find centroid of structures present in segmentation.

    Args:
        seg (np.ndarray): Segmentation of structures.

    Returns:
        centroid_x (int): x-coordinate of centroid.
        centroid_y (int): y-coordinate of centroid.
    """
    # Find the contours of the structures in segmentation
    contours = find_contours(seg)

    # Check if a contour is present. If not, set centroid to NaN.
    if len(contours) == 0:
        centroid_x, centroid_y = np.nan, np.nan
    else:
        # Find largest contour in segmentation.
        main_contour = find_largest_contour(contours)

        # Find moment and middle coordinates of largest contour.
        moments_main_contour = cv2.moments(main_contour)
        centroid_x = int(moments_main_contour["m10"] / moments_main_contour["m00"])
        centroid_y = int(moments_main_contour["m01"] / moments_main_contour["m00"])

    return centroid_x, centroid_y


def find_centroids_of_all_structures(centroids: dict[str, np.ndarray], seg: np.ndarray) -> dict[str, np.ndarray]:
    """Find centroid of all structures present in segmentation and add to dictionary.

    Args:
        centroids (dict[str, np.ndarray]): Initial dictionary with structure numbers as keys and lists of x- and y-coordinates as values.
        seg (np.ndarray): Segmentation of structures.

    Returns:
        centroids (dict[str, np.ndarray]): Extended dictionary with structure numbers as keys and lists of x- and y-coordinates as values.
    """
    # Separate the segmentations, each with its own structures.
    seg_0, seg_1, seg_2, seg_3 = separate_segmentation(seg)

    # Get middle coordinate of each separate segmentation and add to dictionary.
    for structure in centroids:
        centroid_x, centroid_y = find_centroid(locals()[f"seg_{structure}"])
        centroids[structure][0].append(centroid_x)
        centroids[structure][1].append(centroid_y)

    return centroids


def get_mean_centroids(path_to_segmentations: str, files_of_view: list[str], frames_to_process: list[int]) -> dict[str, tuple[int, int]]:
    """Get the mean middle point of each structure in all segmentations of one person.

    Args:
        path_to_segmentations (str): Path to folder with segmentations.
        files_of_view (list[str]): List of all images of one person.
        frames_to_process (list[int]): List of frames that are detected as erroneous.

    Returns:
        mean_centroids (dict[str, tuple[int, int]]): Dictionary with structure numbers as keys and tuples of x- and y-coordinates as values.
    """
    # Create dictionary to store centroids of structures.
    centroids = {1: ([], []), 2: ([], []), 3: ([], [])}

    for frame_nr, filename in enumerate(files_of_view):
        # Define file location and load segmentation.
        file_location_seg = os.path.join(path_to_segmentations, filename)
        seg = get_image_array(file_location_seg)

        # Prevent from not selecting a centroid at all when all frames are detected as erroneous.
        if len(frames_to_process) < round(0.95 * len(files_of_view)):
            if frame_nr not in frames_to_process:
                centroids = find_centroids_of_all_structures(centroids, seg)
        else:
            centroids = find_centroids_of_all_structures(centroids, seg)

    # Compute average centroid of each structure.
    mean_centroids = {
        structure: (int(np.nanmean(coords[0])), int(np.nanmean(coords[1])))
        for structure, coords in centroids.items()
    }

    return mean_centroids


def get_main_contour_lv_la(seg: np.ndarray, mean_centroid: tuple[int, int]) -> tuple[np.ndarray, int]:
    """Extract the main contour in LV and LA segmentations.

    The main contour is the contour that contains the mean centroid of the structure based on all frames in the image sequence.
    The contours that were not selected as main contour are removed from the segmentation.

    Args:
        seg (np.ndarray): Segmentation of structures.
        mean_centroid (tuple[int, int]): Tuple of x- and y-coordinates of mean centroid of structure.

    Returns:
        seg_main (np.ndarray): Segmentation of main contour of structure, without redundant contours.
        num_of_contours (int): Number of contours in segmentation.
    """
    # Find the contours of the structures in segmentation.
    contours = find_contours(seg, "external")

    # Select contour based on position in segmentation, only contour which includes average middle coordinate over all frames is taken into account.
    selected_contour = [
        contour
        for contour in contours
        if (
            cv2.pointPolygonTest(contour, mean_centroid, False) >= 0
            or cv2.pointPolygonTest(
                contour, (mean_centroid[0] - 25, mean_centroid[1]), False
            )
            >= 0
            or cv2.pointPolygonTest(
                contour, (mean_centroid[0] + 25, mean_centroid[1]), False
            )
            >= 0
            or cv2.pointPolygonTest(
                contour, (mean_centroid[0], mean_centroid[1] - 25), False
            )
            >= 0
            or cv2.pointPolygonTest(
                contour, (mean_centroid[0], mean_centroid[1] + 25), False
            )
            >= 0
        )
    ]

    # Initialise seg_main with array of zeros with same size as seg.
    seg_main = np.zeros_like(seg.copy())
    num_of_contours = 0

    # Check if a contours is selected and draw this contour in seg_main.
    if len(selected_contour) > 0:
        cv2.drawContours(seg_main, selected_contour, -1, 1, thickness=-1)
        num_of_contours = 1

    return seg_main, num_of_contours


def get_main_contour_myo(seg_1: np.ndarray, seg_2: np.ndarray, threshold_distance: int = 5) -> np.ndarray:
    """Extract the contours of the myocardium that neighbour the LV.

    Args:
        seg_1 (np.ndarray): Segmentation of LV.
        seg_2 (np.ndarray): Segmentation of MYO.
        threshold_distance (int): Threshold for maximum distance from contour 2 to contour 1 (default: 5).

    Returns:
        seg_main (np.ndarray): Segmentation of main contour of MYO, without redundant contours.
    """
    # Find the contours of the structures in LV and MYO segmentation.
    contour_1 = find_contours(seg_1, "external")
    contours_2 = find_contours(seg_2, "external")

    neighboring_contours = []

    # Select the contours of the MYO that are within a certain distance of the LV contour.
    for i, contour_2 in enumerate(contours_2):
        min_distances = []

        # Calculate the minimum distance between each point of contour 1 and contour 2.
        for j in range(len(contour_1[0])):
            coordinates_1 = tuple(
                [int(contour_1[0][j][0][0]), int(contour_1[0][j][0][1])]
            )
            min_distance = cv2.pointPolygonTest(contour_2, coordinates_1, True)
            min_distances.append(min_distance)

        # Calculate absolute minimum distances.
        abs_distances = [abs(ele) for ele in min_distances]

        # Add the contour to the segmentation if the minimum absolute distance is smaller than the threshold distance.
        if min(abs_distances) < threshold_distance:
            neighboring_contours.append(i)

    # Convert selected MYO contour(s) back to segmentation.
    seg_main = np.zeros_like(seg_2.copy())
    cv2.drawContours(
        seg_main, [contours_2[k] for k in neighboring_contours], -1, 1, thickness=-1
    )

    # Close the segmentation to fill the (small) holes between the contours if multiple contours are selected.
    if len(neighboring_contours) > 1:
        seg_main = cv2.morphologyEx(
            seg_main, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8)
        )

    return seg_main


def fill_holes_within_structure(seg: np.ndarray, label: int) -> np.ndarray:
    """Fill the holes within a structure.

    Args:
        seg (np.ndarray): Segmentation of structures.
        label (int): Label of structure.

    Returns:
        seg_no_holes (np.ndarray): Segmentation of structure with filled holes.
    """
    # Find coordinates of holes within structure.
    coordinates_holes = find_coordinates_of_holes(seg)

    seg_no_holes = seg.copy()

    # Fill the holes with in the structure.
    for row, col in zip(coordinates_holes[0], coordinates_holes[1]):
        seg_no_holes[row, col] = label

    return seg_no_holes


def find_coordinates_of_structure(seg: np.ndarray, label: int) -> list[tuple[int, int]]:
    """Find the coordinates of all pixels in a structure.

    Args:
        seg (np.ndarray): Segmentation of structures.
        label (int): Label of structure.

    Returns:
        coordinates (list[tuple[int, int]]): List of tuples of x- and y-coordinates of pixels in structure.
    """
    indices = np.argwhere(seg == label)
    coordinates = [(x, y) for y, x in indices]

    return coordinates


def fill_holes_between_lv_la(seg: np.ndarray) -> np.ndarray:
    """Fill the holes between LV and LA segmentations.

    Args:
        seg (np.ndarray): Segmentation of structures.

    Returns:
        seg_combined (np.ndarray): Segmentation of structures with filled holes between LV and LA.
    """
    # Separate the segmentations, and combine the LV and MYO, and LV and LA segmentations.
    _, seg_1, seg_2, seg_3 = separate_segmentation(seg)
    seg_12 = combine_segmentations([seg_1, seg_2], "difference", [1, 2])
    seg_13 = combine_segmentations([seg_1, seg_3], "difference", [1, 3])

    # Find the coordinates of holes between seg1/2 and seg2/3.
    coordinates_holes_12 = find_coordinates_of_holes(seg_12)
    coordinates_holes_13 = find_coordinates_of_holes(seg_13)

    # Find coordinates of LV.
    coordinates_seg_1 = find_coordinates_of_structure(seg_1, 1)

    seg_13_no_holes = seg_13.copy()

    # Fill holes between LV and LA.
    for row, col in zip(coordinates_holes_13[0], coordinates_holes_13[1]):
        if (col, row) not in coordinates_seg_1 and (row, col) not in list(
            zip(coordinates_holes_12[0], coordinates_holes_12[1])
        ):
            # Extract the neighborhood of the pixel to change.
            neighborhood = seg_13_no_holes[row - 1 : row + 1, col - 1 : col + 1]

            # Count number of MYO and LA pixels in neighborhood.
            num_lv_px = np.count_nonzero(neighborhood == 1)
            num_la_px = np.count_nonzero(neighborhood == 3)

            # Calculate the mean value of the neighborhood.
            new_pixel_value = 1 if num_lv_px >= num_la_px else 3

            # Change the pixel value in the original image.
            seg_13_no_holes[row, col] = new_pixel_value

    # Combine the segmentations of all structures.
    seg_combined = seg_2 + seg_13_no_holes

    return seg_combined


def fill_holes_between_myo_la(seg_total: np.ndarray) -> np.ndarray:
    """Fill the holes between MYO and LA segmentations.

    Args:
        seg_total (np.ndarray): Segmentation of structures.

    Returns:
        seg_combined (np.ndarray): Segmentation of structures with filled holes between MYO and LA.
    """
    # Separate the segmentations, and combine the LV and MYO, and LV and LA segmentations.
    _, seg_1, seg_2, seg_3 = separate_segmentation(seg_total)
    seg_12 = combine_segmentations([seg_1, seg_2], "difference", [1, 2])
    seg_23 = combine_segmentations([seg_2, seg_3], "difference", [2, 3])

    # Find coordinates of the holes between seg1/2 and seg2/3.
    coordinates_holes_12 = find_coordinates_of_holes(seg_12)
    coordinates_holes_23 = find_coordinates_of_holes(seg_23)

    # Find coordinates of LV.
    coordinates_seg_1 = find_coordinates_of_structure(seg_1, 1)

    seg_23_no_holes = seg_23

    # Fill holes between MYO and LA.
    for row, col in zip(coordinates_holes_23[0], coordinates_holes_23[1]):

        if (col, row) not in coordinates_seg_1 and (row, col) not in list(
            zip(coordinates_holes_12[0], coordinates_holes_12[1])
        ):
            # Extract the neighborhood of the pixel to change.
            neighborhood = seg_23_no_holes[row - 1 : row + 1, col - 1 : col + 1]

            # Count number of MYO and LA pixels in neighborhood.
            num_myo_px = np.count_nonzero(neighborhood == 2)
            num_la_px = np.count_nonzero(neighborhood == 3)

            # Calculate the mean value of the neighborhood.
            new_pixel_value = 2 if num_myo_px >= num_la_px else 3

            # Change the pixel value in the original image.
            seg_23_no_holes[row, col] = new_pixel_value

    # Combine the segmentations of all structures.
    seg_combined = seg_1 + seg_23_no_holes

    return seg_combined


def find_border_pixels_holes(distances: list[float]) -> list[tuple[int, int]]:
    """Find the start and end indices of the holes between the structures.

    Args:
        distances (list[float]): List of distances between the contours of the structures.

    Returns:
        indices (list[tuple[int, int]]): List of tuples of start and end indices of the holes between the structures.
    """
    indices = []
    start_idx = None

    for i, value in enumerate(distances):
        if value > 1:
            if start_idx is None:
                start_idx = i
        elif value == 1 and start_idx is not None:
            indices.append((start_idx, i - 1))
            start_idx = None

    # Check if the last element is not equal to 1. If so, add 1 to the end index.
    if start_idx is not None and distances[-1] > 1:
        indices.append((start_idx, len(distances) - 1))

    return indices


def fill_holes_between_lv_myo(seg_total: np.ndarray) -> np.ndarray:
    """Fill the holes between LV and MYO segmentations.

    Args:
        seg_total (np.ndarray): Segmentation of structures.

    Returns:
        seg_filled (np.ndarray): Segmentation of structures with filled holes between LV and MYO.
    """
    # Separate the segmentations, and combine the LV and MYO. segmentations.
    _, seg_1, seg_2, seg_3 = separate_segmentation(seg_total)
    total_seg_12 = combine_segmentations([seg_1, seg_2], "difference")

    # Find contours of LV, MYO and LA.
    contour_1 = find_contours(seg_1, "external")
    contour_2 = find_contours(seg_2, "external")
    contour_3 = find_contours(seg_3, "external")

    # Check for presence of contours for MYO and LA.
    if len(contour_2) > 0 and len(contour_3) > 0:
        min_distances_12 = [
            abs(cv2.pointPolygonTest(contour_2[0], tuple(map(int, point[0])), True))
            for point in contour_1[0]
        ]
        min_distances_13 = [
            abs(cv2.pointPolygonTest(contour_3[0], tuple(map(int, point[0])), True))
            for point in contour_1[0]
        ]

        # Check if the distance between LV and MYO and LV and LA are both smaller or equal to 1.
        if min(min_distances_12) <= 1.0 and min(min_distances_13) <= 1.0:
            # Find the minimum distance between the border of contour 1 and both other contours.
            distances = [min(pair) for pair in zip(min_distances_12, min_distances_13)]

            # Find coordinates of the holes.
            holes = find_border_pixels_holes(distances)

            for hole in holes:
                # Define coordinates of start points of holes.
                coordinates_x_1, coordinates_y_1 = (
                    contour_1[0][hole[0]][0][0],
                    contour_1[0][hole[0]][0][1],
                )
                coordinates_x_2, coordinates_y_2 = (
                    contour_1[0][hole[1]][0][0],
                    contour_1[0][hole[1]][0][1],
                )

                # Create list with indices where contour 2 neighbors contour 1.
                list_j = [
                    j
                    for j, coor in enumerate(contour_2[0])
                    if abs(
                        cv2.pointPolygonTest(
                            contour_1[0],
                            tuple([int(coor[0][0]), int(coor[0][1])]),
                            True,
                        )
                    )
                    <= 1
                ]

                # Final all points on contour of myocardium and remove all inside points.
                contour_points = np.array(contour_2[0])[:, 0].reshape(-1, 2)
                outside_c_points_x = list(contour_points[:, 0][: list_j[0]]) + list(
                    contour_points[:, 0][list_j[-1] :]
                )
                outside_c_points_y = list(contour_points[:, 1][: list_j[0]]) + list(
                    contour_points[:, 1][list_j[-1] :]
                )

                # Calculate the minimum difference between the gap points to the outside of the myocardium.
                distances_gap_1 = [
                    np.sqrt((x - coordinates_x_1) ** 2 + (y - coordinates_y_1) ** 2)
                    for x, y in zip(outside_c_points_x, outside_c_points_y)
                ]
                distances_gap_2 = [
                    np.sqrt((x - coordinates_x_2) ** 2 + (y - coordinates_y_2) ** 2)
                    for x, y in zip(outside_c_points_x, outside_c_points_y)
                ]
                min_myo_thickness = min(min(distances_gap_1), min(distances_gap_2))

                # Loop over all pixels with value 0, check if difference between pixel and outside border of myocardium
                # (epicardium) is smaller than the minimum myocardial thickness. If this is the case, give pixel value 2.
                for row, col in zip(zero_positions[0], zero_positions[1]):
                    for x, y in zip(outside_c_points_x, outside_c_points_y):
                        if (
                            np.sqrt((x - col) ** 2 + (y - row) ** 2)
                            <= min_myo_thickness
                        ):
                            total_seg_12[row, col] = 2

        # Recheck for pixels within structure with value 0. If this is still the case, assign pixel value 1 to pixel.
        zero_positions = find_coordinates_of_holes(total_seg_12)
        for row, col in zip(zero_positions[0], zero_positions[1]):
            total_seg_12[row, col] = 1

    # Combine the segmentations of all structures.
    seg_filled = total_seg_12 + seg_3

    return seg_filled


def post_process_segmentation(seg: np.ndarray, centroids: dict[int, tuple[int, int]]) -> np.ndarray:
    """Post-process segmentation.

    Post-processing consists of the following steps:
        1. Select main contours of each structure, if multiple present.
        2. Fill holes within each structure.
        3. Fill holes between structures.

    Args:
        seg (np.ndarray): Segmentation of structures.
        centroids (dict[int, tuple[int, int]]): Dictionary with structure numbers as keys and tuples of x- and y-coordinates as values.

    Returns:
        seg_total (np.ndarray): Post-processed segmentation of structures.
    """
    # Separate the segmentations, each with its own structures.
    (
        _,
        seg_1_before_processing,
        seg_2_before_processing,
        seg_3_before_processing,
    ) = separate_segmentation(seg)

    # Fill the holes within each structure, this is to fill any holes in any structure before the main contours are found.
    seg_1_before_processing = fill_holes_within_structure(seg_1_before_processing, 1)
    seg_3_before_processing = fill_holes_within_structure(seg_3_before_processing, 3)

    # Get separate segmentation and number of contours in the LV and LA.
    seg_1, num_of_contours1 = get_main_contour_lv_la(
        seg_1_before_processing, centroids[1]
    )
    seg_3, _ = get_main_contour_lv_la(seg_3_before_processing, centroids[3])

    # If LV contour is present, use this to find main MYO contour(s).
    seg_2 = (
        get_main_contour_myo(seg_1, seg_2_before_processing)
        if num_of_contours1 > 0
        else seg_2_before_processing
    )

    # Fill the holes within each structure.
    seg_1 = fill_holes_within_structure(seg_1, 1)
    seg_2 = fill_holes_within_structure(seg_2, 2)
    seg_3 = fill_holes_within_structure(seg_3, 3)

    # Combine segmentations between the structures, with different labels for each structure.
    seg_total = combine_segmentations(
        [seg_1, seg_2, seg_3], "difference_with_overlap", [1, 2, 3]
    )

    # Separate the segmentations, each with its own structures.
    _, seg_1, seg_2, seg_3 = separate_segmentation(seg_total)

    # Combine segmentations of LV and MYO, and LV and LA, with a different label for each structure.
    seg_12 = combine_segmentations([seg_1, seg_2], "difference", [1, 2])
    seg_13 = combine_segmentations([seg_1, seg_3], "difference", [1, 3])

    # Check for holes between LV and MYO, and fill them if present.
    if len(find_coordinates_of_holes(seg_12)[0]) > 0:
        seg_total = fill_holes_between_lv_myo(seg_total)

    # Check for holes between LV and LA, and fill them if present.
    if len(find_coordinates_of_holes(seg_13)[0]) > 0:
        seg_total = fill_holes_between_lv_la(seg_total)

    # Check for holes between MYO and LA, and fill them if present.
    if len(find_coordinates_of_holes(seg_total)[0]) > 0:
        seg_total = fill_holes_between_myo_la(seg_total)

    return seg_total


def main_post_processing(
    path_to_segmentations: str,
    path_to_final_segmentations: str,
    single_frame_qc: dict[str, list[int]],
    all_files: list[str],
    views: list[str],
) -> None:
    """Main function to do post-processing of all segmentations.

    The post-processed segmentations will be saved in the folder specified by path_to_final_segmentations.

    Args:
        path_to_segmentations (str): Path to folder with segmentations.
        path_to_final_segmentations (str): Path to folder where post-processed segmentations should be saved.
        single_frame_QC (dict[str, list[int]]): Dictionary with patient IDs as keys and lists of frames that are detected as erroneous as values.
        all_files (list[str]): List of all files in the directory.
        views (list[str]): List of views of the segmentations.
    """
    for view in views:
        # Get all files of one view of one person.
        files_of_view = get_list_with_files_of_view(all_files, view)

        # Get the frames that need to be processed.
        frames_to_process = single_frame_qc["flagged_frames"][view]

        # Get the mean centroids of all structures from all segmentations of one person.
        centroids = get_mean_centroids(
            path_to_segmentations, files_of_view, frames_to_process
        )

        for frame_nr, filename in enumerate(files_of_view):
            # In case post-processing is needed, do post-processing and save segmentation in final folder.
            if frame_nr in frames_to_process:
                # Define file location and load segmentation.
                file_location_seg = os.path.join(path_to_segmentations, filename)
                initial_segmentation = get_image_array(file_location_seg)

                # Post-process segmentation.
                seg_total = post_process_segmentation(initial_segmentation, centroids)

                # Save post-processed segmentation.
                itk_seg_after_pp = sitk.GetImageFromArray(seg_total)
                save_path = os.path.join(path_to_final_segmentations, filename)
                sitk.WriteImage(itk_seg_after_pp, save_path)

            # In case no post-processing needed, copy segmentation to final folder.
            else:
                segmentation_location = os.path.join(path_to_segmentations, filename)
                shutil.copy(segmentation_location, path_to_final_segmentations)
