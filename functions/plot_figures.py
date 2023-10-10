# This script contains functions to create figures.
import os
import cv2
import numpy as np
from PIL import Image
from typing import Union
import matplotlib.pyplot as plt
from functions.general_utilities import *


def _color_segmentation(seg: np.ndarray, colors_for_labels: np.ndarray) -> np.ndarray:
    """Give each label in a segmentation a certain color.

    Args:
        seg (np.ndarray): Segmentation to be colored.
        colors_for_labels (np.ndarray): Color definitions for each label.

    Returns:
        seg_colored (np.ndarray): Colored segmentation.
    """
    seg_colored = colors_for_labels[seg]

    return seg_colored


def _remove_neighboring_pixels(
    contours_1: list[np.ndarray],
    contours_2: list[np.ndarray],
    distance_threshold: int = 3,
) -> list[np.ndarray]:
    """Remove points on the contours that neighbor the left ventricular (LV) contour.

    Args:
        contours_1 (list[np.ndarray]): List of contours of the LV.
        contours_2 (list[np.ndarray]): List of contours of the myocardium (MYO) or left atrium (LA).
        threshold_distance (int): Threshold distance between the LV and MYO or LA contour to remove points on the MYO or LA contour (default: 3).

    Returns:
        contour_adapted (list[np.ndarray]): List of contours of the MYO or LA with points removed.
    """
    if len(contours_1) > 0 and len(contours_2) > 0:
        contour_adapted = []

        # Loop over all points on the contours of the MYO or LA to check if they neighbor the LV contour.
        for contour_2 in contours_2:
            for j in range(len(contour_2)):
                x_coordinate_2 = int(contour_2[j][0][0])
                y_coordinate_2 = int(contour_2[j][0][1])

                coordinates_2 = tuple([x_coordinate_2, y_coordinate_2])

                min_distance = cv2.pointPolygonTest(contours_1[0], coordinates_2, True)

                # If the distance between the point on the MYO or LA contour and the LV contour is larger than x pixels, add the point to the list of points.
                if abs(min_distance) > distance_threshold:
                    contour_adapted.append([x_coordinate_2, y_coordinate_2])

        # Convert list of points to a numpy array and reshape to format of original contours.
        contour_adapted = np.array(contour_adapted)
        contour_adapted = tuple(
            contour_adapted.reshape((contour_adapted.shape[0], 1, 2))[np.newaxis, :]
        )

    else:
        contour_adapted = contours_2

    return contour_adapted


def _project_segmentation_on_image(
    image: np.ndarray, seg: np.ndarray, label_colors: np.ndarray
) -> np.ndarray:
    """Project contours of a segmentation on the corresponding echo image.

    Args:
        image (np.ndarray): Image to project contours on.
        seg (np.ndarray): Segmentation to project contours of.
        label_colors (np.ndarray): Color definitions for each label.

    Returns:
        image_with_contours (np.ndarray): Image with projected contours.
    """
    image_with_contours = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # Map each label value to a specific color.
    colored_labels = _color_segmentation(seg, label_colors)

    # Convert the colored labels back to an image.
    image_for_blend = Image.fromarray(image_with_contours)
    color_labels = Image.fromarray(colored_labels.astype("uint8"))

    # Create blended image.
    overlay_image = Image.blend(image_for_blend, color_labels, 0.1)
    image_with_contours = np.array(overlay_image)

    # Separate contours for each label.
    _, seg_1, seg_2, seg_3 = separate_segmentation(seg)

    contours_1 = find_contours(seg_1, "all")
    contours_2 = find_contours(seg_2, "all")
    contours_3 = find_contours(seg_3, "all")

    # Remove points on the contours that neighbor the LV contour.
    contours_2 = _remove_neighboring_pixels(contours_1, contours_2)
    contours_3 = _remove_neighboring_pixels(contours_1, contours_3)

    # Draw all contours on the image
    for contour_2 in contours_2:
        cv2.drawContours(
            image_with_contours,
            contour_2,
            -1,
            (int(label_colors[2][0]), int(label_colors[2][1]), int(label_colors[2][2])),
            4,
        )

    for contour_3 in contours_3:
        cv2.drawContours(
            image_with_contours,
            contour_3,
            -1,
            (int(label_colors[3][0]), int(label_colors[3][1]), int(label_colors[3][2])),
            4,
        )

    cv2.drawContours(
        image_with_contours,
        contours_1,
        -1,
        (int(label_colors[1][0]), int(label_colors[1][1]), int(label_colors[1][2])),
        4,
    )

    return image_with_contours


def main_plot_area_time_curves(
    path_to_images: str,
    path_to_segmentations: str,
    all_files: list[str],
    views: list[str],
    dicom_properties: dict[str, dict[str, list[float]]],
    segmentation_properties: dict[str, dict[str, Union[list[float], list[int]]]],
    colors_for_labels: np.ndarray,
    font_size: int = 8,
    dpi_value: int = 100,
) -> None:
    """Plot area-time curves for all patients.

    Args:
        path_to_images (str): Directory of the folder with the echo images.
        path_to_segmentations (str): Directory of the folder with the segmentations.
        all_files (list[str]): List of all files in the directory.
        views (list[str]): List of views of the segmentations.
        dicom_properties (dict[str, dict[str, list[int]]]): Dictionary with dicom properties of all patients.
        segmentation_properties (dict[str, dict[str, Union[list[float], list[int]]]]): Dictionary with segmentation properties of all patients.
        colors_for_labels (np.ndarray): Color definitions for each label.
        font_size (int): Font size of the figure (default: 8).
        dpi_value (int): DPI value of the figure (default: 100).
    """
    # Define directories of images and segmentations.
    for view in views:
        # Get dicom and segmentation properties of one patient
        ed_points = segmentation_properties["ed_points"][view]
        es_points = segmentation_properties["es_points"][view]
        lv_areas = segmentation_properties["lv_areas"][view]
        myo_areas = segmentation_properties["myo_areas"][view]
        la_areas = segmentation_properties["la_areas"][view]
        frame_times = dicom_properties["times_frames"][view]

        # Plotting settings
        min_y_val = int(0.9 * min(min(lv_areas), min(myo_areas), min(la_areas)))
        max_y_val = int(1.1 * max(max(lv_areas), max(myo_areas), max(la_areas)))

        # Get all files of one view of one person.
        files_of_view = get_list_with_files_of_view(all_files, view)

        # Plot area-time curves for all frames of one patient.
        # Only plot the first frame for now.
        for frame_nr, filename in enumerate(files_of_view[0:1]):
            # Define file location and load echo image frame.
            file_location_image = define_path_to_images(path_to_images, filename)
            echo_image = convert_image_to_array(file_location_image)

            # Define file location and load segmentation.
            file_location_seg = os.path.join(path_to_segmentations, filename)
            seg = convert_image_to_array(file_location_seg)
            seg_colored = _color_segmentation(seg, colors_for_labels)

            contours_seg = _project_segmentation_on_image(
                echo_image, seg, colors_for_labels
            )

            plt.figure(dpi=dpi_value)
            plt.suptitle(("Segmentation of " + view + ", frame " + str(frame_nr)))

            # Format figure with subplots.
            X = [(2, 3, 1), (2, 3, 2), (2, 3, 3), (2, 3, (4, 5))]

            for nr_plot, (nrows, ncols, plot_number) in enumerate(X):
                if nr_plot == 0:
                    plt.subplot(nrows, ncols, plot_number)
                    plt.imshow(echo_image, cmap="gray")
                    plt.title("Echo image", fontsize=font_size, loc="left")
                    plt.axis("off")

                elif nr_plot == 1:
                    plt.subplot(nrows, ncols, plot_number)
                    plt.imshow(seg_colored)
                    plt.title("Segmentation", fontsize=font_size, loc="left")
                    plt.axis("off")

                elif nr_plot == 2:
                    plt.subplot(nrows, ncols, plot_number)
                    plt.imshow(contours_seg, cmap="gray", interpolation="none")
                    plt.title("Overlay", fontsize=font_size, loc="left")
                    plt.axis("off")

                elif nr_plot == 3:
                    plt.subplot(nrows, ncols, plot_number)
                    plt.title("Area-time curves", fontsize=font_size, loc="left")
                    plt.plot(
                        frame_times, lv_areas, color="green", label="Left Ventricle"
                    )
                    plt.plot(frame_times, myo_areas, color="red", label="Myocardium")
                    plt.plot(frame_times, la_areas, color="blue", label="Left Atrium")

                    # Plot vertical lines for current frame, ES and ED points
                    plt.axvline(
                        x=frame_times[frame_nr],
                        color="c",
                        linewidth=1,
                        linestyle="--",
                        label="Time of frame",
                    )

                    # Plot vertical lines for all ED and ES points
                    plt.axvline(
                        x=frame_times[es_points[0]],
                        color="m",
                        linewidth=1,
                        linestyle="--",
                        label="ES point",
                    )
                    plt.axvline(
                        x=frame_times[ed_points[0]],
                        color="y",
                        linewidth=1,
                        linestyle="--",
                        label="ED point",
                    )
                    for idx in range(1, len(es_points)):
                        plt.axvline(
                            x=frame_times[es_points[idx]],
                            color="m",
                            linewidth=1,
                            linestyle="--",
                        )
                    for idx in range(1, len(ed_points)):
                        plt.axvline(
                            x=frame_times[ed_points[idx]],
                            color="y",
                            linewidth=1,
                            linestyle="--",
                        )

                    plt.xlabel("Time [ms]")
                    plt.ylabel("Area [cm$^{2}$]")
                    plt.xlim(frame_times[0] - 40, frame_times[-1] + 40)
                    plt.ylim(min_y_val, max_y_val)
                    plt.legend(bbox_to_anchor=(1, 0.5), loc="center left")


def show_atlases(
    atlas_lv: list[float], atlas_la: list[float], dpi_value: int = 100
) -> None:
    """Function to plot the atlases (population priors) for the LV and LA.

    Args:
        atlas_lv (list[float]): Atlas of the LV.
        atlas_la (list[float]): Atlas of the LA.
        dpi_value (int): DPI value of the figure (default: 100).
    """
    plt.figure(dpi=dpi_value)
    plt.title("Atlases LV and LA area-time curves")
    plt.plot(atlas_lv, color=(0, 1, 0), linewidth=5)
    plt.plot(atlas_la, color=(0, 0, 1), linewidth=5)
    plt.legend(["LV", "LA"])
    plt.xlabel("% of a cardiac cycle")
    plt.ylabel("Normalised area [-]")


def show_post_processing_results(
    path_to_images: str,
    path_to_segmentations: str,
    path_to_final_segmentations: str,
    all_files: list[str],
    views: list[str],
    single_frame_qc: dict[
        str, Union[dict[str, list[bool]], dict[str, list[int]], dict[str, int]]
    ],
    colors_for_labels: np.ndarray,
) -> None:
    """Function to plot the results of the post-processing.

    Args:
        path_to_images (str): Directory of the folder with the echo images.
        path_to_segmentations (str): Directory of the folder with the segmentations.
        path_to_final_segmentations (str): Directory of the folder with the final segmentations.
        all_files (list[str]): List of all files in the directory.
        views (list[str]): List of views of the segmentations.
        single_frame_qc (dict[str, Union[dict[str, list[bool]], dict[str, list[int]], dict[str, int]]]): Dictionary with the single frame QC results.
        colors_for_labels (np.ndarray): Color definitions for each label.
    """
    for view in views:
        # Get the frames that need to be processed.
        frames_to_process = single_frame_qc["flagged_frames"][view]

        if len(frames_to_process) == 0:
            print("No frames were processed for view " + view + ".")

        # Get all files of one view of one person.
        files_of_view = get_list_with_files_of_view(all_files, view)

        for frame_nr, filename in enumerate(files_of_view):
            if frame_nr in frames_to_process:
                # Define file locations and load images and segmentations.
                file_location_image = define_path_to_images(path_to_images, filename)
                echo_image = convert_image_to_array(file_location_image)

                file_location_seg_before_pp = os.path.join(
                    path_to_segmentations, filename
                )
                seg_before_pp = _color_segmentation(
                    convert_image_to_array(file_location_seg_before_pp),
                    colors_for_labels,
                )

                file_location_seg_after_pp = os.path.join(
                    path_to_final_segmentations, filename
                )
                seg_after_pp = _color_segmentation(
                    convert_image_to_array(file_location_seg_after_pp),
                    colors_for_labels,
                )

                plt.figure(figsize=(15, 5))
                plt.suptitle(("Segmentation of " + view + ", frame " + str(frame_nr)))

                # Subplot 1: Echo image.
                plt.subplot(1, 3, 1)
                plt.imshow(echo_image, cmap="gray")
                plt.title("Echo image")
                plt.axis("off")

                # Subplot 2: Segmentation before post-processing.
                plt.subplot(1, 3, 2)
                plt.imshow(seg_before_pp)
                plt.title("Segmentation before post-processing")
                plt.axis("off")

                # Subplot 3: Segmentation after post-processing.
                plt.subplot(1, 3, 3)
                plt.imshow(seg_after_pp)
                plt.title("Segmentation after post-processing")
                plt.axis("off")
