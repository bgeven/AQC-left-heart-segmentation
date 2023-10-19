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
        contours_1 (list[np.ndarray]): Contour(s) of the LV.
        contours_2 (list[np.ndarray]): Contour(s) of the myocardium (MYO) or left atrium (LA).
        threshold_distance (int): Threshold distance between the LV and MYO or LA contour to remove points on the MYO or LA contour (default: 3).

    Returns:
        contour_adapted (list[np.ndarray]): Contour(s) of the MYO or LA with points removed.
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
    segmentation_properties: dict[str, dict[str, list[float]]],
    colors_for_labels: np.ndarray,
    font_size: int = 8,
    dpi_value: int = 100,
) -> plt.figure:
    """Plot area-time curves for all patients.

    Args:
        path_to_images (str): Path to the directory containing the echo images.
        path_to_segmentations (str): Path to the directory containing the segmentations.
        all_files (list[str]): All files in the directory.
        views (list[str]): Plane views of the segmentations.
        dicom_properties (dict[str, dict[str, list[float]]]): Dictionary containing the properties of the DICOM files.
        segmentation_properties (dict[str, dict[str, list[float]]]): Dictionary containing the segmentation parameters.
        colors_for_labels (np.ndarray): Color definitions for each label.
        font_size (int): Font size of the figure (default: 8).
        dpi_value (int): DPI value of the figure (default: 100).

    Returns:
        fig (list[plt.figure]): Figures with image, segmentation and area-time curves for each image.
    """
    figures = []

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

            fig = plt.figure(dpi=dpi_value)
            fig.suptitle(
                "Segmentation and area-time curves. View: {}, frame {}.".format(
                    view, str(frame_nr)
                )
            )

            # Format figure with subplots.
            X = [(2, 3, 1), (2, 3, 2), (2, 3, 3), (2, 3, (4, 5))]

            for nr_plot, (nrows, ncols, plot_number) in enumerate(X):
                ax = fig.add_subplot(nrows, ncols, plot_number)

                if nr_plot == 0:
                    ax.imshow(echo_image, cmap="gray")
                    ax.set_title("Echo image", fontsize=font_size, loc="left")
                    ax.axis("off")

                elif nr_plot == 1:
                    ax.imshow(seg_colored)
                    ax.set_title("Segmentation", fontsize=font_size, loc="left")
                    ax.axis("off")

                elif nr_plot == 2:
                    ax.imshow(contours_seg, cmap="gray", interpolation="none")
                    ax.set_title("Overlay", fontsize=font_size, loc="left")
                    ax.axis("off")

                elif nr_plot == 3:
                    ax.set_title("Area-time curves", fontsize=font_size, loc="left")
                    ax.plot(
                        frame_times, lv_areas, color="green", label="Left Ventricle"
                    )
                    ax.plot(frame_times, myo_areas, color="red", label="Myocardium")
                    ax.plot(frame_times, la_areas, color="blue", label="Left Atrium")

                    # Plot vertical lines for current frame, ES and ED points
                    ax.axvline(
                        x=frame_times[frame_nr],
                        color="c",
                        linewidth=1,
                        linestyle="--",
                        label="Time of frame",
                    )

                    # Plot vertical lines for all ED and ES points
                    ax.axvline(
                        x=frame_times[es_points[0]],
                        color="m",
                        linewidth=1,
                        linestyle="--",
                        label="ES point",
                    )
                    ax.axvline(
                        x=frame_times[ed_points[0]],
                        color="y",
                        linewidth=1,
                        linestyle="--",
                        label="ED point",
                    )
                    for idx in range(1, len(es_points)):
                        ax.axvline(
                            x=frame_times[es_points[idx]],
                            color="m",
                            linewidth=1,
                            linestyle="--",
                        )
                    for idx in range(1, len(ed_points)):
                        ax.axvline(
                            x=frame_times[ed_points[idx]],
                            color="y",
                            linewidth=1,
                            linestyle="--",
                        )

                    ax.set_xlabel("Time [ms]")
                    ax.set_ylabel("Area [cm$^{2}$]")
                    ax.set_xlim(frame_times[0] - 40, frame_times[-1] + 40)
                    ax.set_ylim(min_y_val, max_y_val)
                    ax.legend(bbox_to_anchor=(1, 0.5), loc="center left")

            # Append the current figure to the list
            figures.append(fig)

    return figures


def alt_plot_area_time_curves(
    views: list[str],
    segmentation_properties: dict[str, dict[str, list[float]]],
    dpi_value: int = 100,
) -> plt.figure:
    """Plot area-time curves for all patients.

    Args:
        views (list[str]): Plane views of the segmentations.
        segmentation_properties (dict[str, dict[str, list[float]]]): Dictionary containing the segmentation parameters.
        dpi_value (int): DPI value of the figure (default: 100).

    Returns:
        fig (list[plt.figure]): Figures with area-time curves for each image.
    """
    figures = []

    # Define directories of images and segmentations.
    for view in views:
        # Get dicom and segmentation properties of one patient
        ed_points = segmentation_properties["ed_points"][view]
        es_points = segmentation_properties["es_points"][view]
        lv_areas = segmentation_properties["lv_areas"][view]
        myo_areas = segmentation_properties["myo_areas"][view]
        la_areas = segmentation_properties["la_areas"][view]

        # Plotting settings
        min_y_val = int(0.9 * min(min(lv_areas), min(myo_areas), min(la_areas)))
        max_y_val = int(1.1 * max(max(lv_areas), max(myo_areas), max(la_areas)))

        # Plot area-time curves for all frames of one patient.
        fig, ax = plt.subplots(dpi=dpi_value)
        fig.suptitle("Area-time curves. View: {}.".format(view))

        ax.plot(lv_areas, color="green", label="Left Ventricle")
        ax.plot(myo_areas, color="red", label="Myocardium")
        ax.plot(la_areas, color="blue", label="Left Atrium")

        # Plot vertical lines for all ED and ES points
        ax.axvline(
            x=es_points[0],
            color="m",
            linewidth=1,
            linestyle="--",
            label="ES point",
        )
        ax.axvline(
            x=ed_points[0],
            color="y",
            linewidth=1,
            linestyle="--",
            label="ED point",
        )
        for idx in range(1, len(es_points)):
            ax.axvline(
                x=es_points[idx],
                color="m",
                linewidth=1,
                linestyle="--",
            )
        for idx in range(1, len(ed_points)):
            ax.axvline(
                x=ed_points[idx],
                color="y",
                linewidth=1,
                linestyle="--",
            )

        ax.set_xlabel("Frame number [-]")
        ax.set_ylabel("Area [cm$^{2}$]")
        ax.set_ylim(min_y_val, max_y_val)
        ax.legend(bbox_to_anchor=(1, 0.5), loc="center left")

        # Append the current figure to the list
        figures.append(fig)

    return figures


def show_atlases(
    atlas_lv: list[float], atlas_la: list[float], dpi_value: int = 100
) -> tuple[plt.figure, plt.axes]:
    """Function to plot the atlases (population priors) for the LV and LA.

    Args:
        atlas_lv (list[float]): Values of the area-time curve of the atlas for the LV.
        atlas_la (list[float]): Values of the area-time curve of the atlas for the LA.
        dpi_value (int): DPI value of the figure (default: 100).

    Returns:
        fig (plt.figure): Figure with the atlases.
        ax (plt.axes): Axes of the figure.
    """
    fig, ax = plt.subplots(dpi=dpi_value)
    ax.set_title("Atlases LV and LA area-time curves")
    ax.plot(atlas_lv, color=(0, 1, 0), linewidth=5)
    ax.plot(atlas_la, color=(0, 0, 1), linewidth=5)
    ax.legend(["LV", "LA"])
    ax.set_xlabel("% of a cardiac cycle")
    ax.set_ylabel("Normalised area [-]")

    return fig, ax


def show_post_processing_results(
    path_to_images: str,
    path_to_segmentations: str,
    path_to_final_segmentations: str,
    all_files: list[str],
    views: list[str],
    single_frame_qc: dict[str, dict[str, list]],
    colors_for_labels: np.ndarray,
) -> tuple[plt.figure, plt.axes, plt.axes, plt.axes]:
    """Function to plot the results of the post-processing.

    Args:
        path_to_images (str): Path to the directory containing the echo images.
        path_to_segmentations (str): Path to the directory containing the segmentations.
        path_to_final_segmentations (str): Path to the directory containing the final segmentations.
        all_files (list[str]): All files in the directory.
        views (list[str]): Plane views of the segmentations.
        single_frame_qc (dict[str, dict[str, list]]): Dictionary containing the results of the single-frame QC.
        colors_for_labels (np.ndarray): Color definitions for each label.

    Returns:
        fig (list[plt.figure]): Figures with the results of the post-processing.
    """
    figures = []

    for view in views:
        # Get the frames that need to be processed.
        frames_to_process = single_frame_qc["flagged_frames"][view]

        if len(frames_to_process) == 0:
            print("No frames were processed for this view")
            # print("No frames were processed for view " + view + ".")

        # Get all files of one view of one person.
        files_of_view = get_list_with_files_of_view(all_files, view)

        for frame_nr, filename in enumerate(files_of_view):
            if frame_nr in frames_to_process:
                # Define file locations and load images and segmentations.
                images_present = len(os.listdir(path_to_images)) > 0
                if images_present == True:
                    file_location_image = define_path_to_images(
                        path_to_images, filename
                    )
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

                fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
                fig.suptitle(
                    "Visualisation post-processing. View: {}, frame: {}.".format(
                        view, str(frame_nr)
                    )
                )

                # Subplot 1: Echo image.
                if images_present == True:
                    ax1.imshow(echo_image, cmap="gray")
                    ax1.set_title("Echo image")
                    ax1.axis("off")
                else:
                    ax1.axis("off")
                    ax1.text(
                        0.5,
                        0.5,
                        "No image available.",
                        clip_on=True,
                        horizontalalignment="center",
                        verticalalignment="center",
                    )

                # Subplot 2: Segmentation before post-processing.
                ax2.imshow(seg_before_pp)
                ax2.set_title("Segmentation before post-processing")
                ax2.axis("off")

                # Subplot 3: Segmentation after post-processing.
                ax3.imshow(seg_after_pp)
                ax3.set_title("Segmentation after post-processing")
                ax3.axis("off")

                # Append the current figure to the list
                figures.append(fig)

    return figures
