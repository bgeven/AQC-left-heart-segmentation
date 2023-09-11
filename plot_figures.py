# This script contains functions to create figures.
import os
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from general_utilities import get_image_array, separate_segmentation, find_contours, get_list_with_files_of_view


def color_segmentation(seg, colors_for_labels):
    """Function to color a segmentation with colors for each label.

    Args:
        seg (np.ndarray): Segmentation to be colored.
        colors_for_labels (np.ndarray): Color definitions for each label.

    Returns:
        seg_colored (np.ndarray): Colored segmentation.
    """
    seg_colored = colors_for_labels[seg]

    return seg_colored


def remove_neighboring_pixels(contours_1, contours_2, threshold_distance=3):
    """Function to remove points on the contours that neighbor the LV contour.

    Args:
        contours_1 (list): List of contours of the LV.
        contours_2 (list): List of contours of the MYO or LA.
        threshold_distance (int): Threshold distance between the LV and MYO or LA contour to remove points on the MYO or LA contour (default: 3).

    Returns:
        contour_adapted (list): List of contours of the MYO or LA with points removed.
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
                if abs(min_distance) > threshold_distance:
                    contour_adapted.append([x_coordinate_2, y_coordinate_2])

        # Convert list of points to a numpy array and reshape to format of original contours.
        contour_adapted = np.array(contour_adapted)
        contour_adapted = tuple(
            contour_adapted.reshape((contour_adapted.shape[0], 1, 2))[np.newaxis, :]
        )

    else:
        contour_adapted = contours_2

    return contour_adapted


def color_contours_segmentation(image, seg, label_colors):
    """Function to project contours of a segmentation on an image.

    Args:
        image (np.ndarray): Image to project contours on.
        seg (np.ndarray): Segmentation to project contours of.
        label_colors (np.ndarray): Color definitions for each label.

    Returns:
        image_with_contours (np.ndarray): Image with projected contours.
    """
    image_with_contours = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # Map each label value to a specific color. 
    colored_labels = color_segmentation(seg, label_colors)

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
    contours_2 = remove_neighboring_pixels(contours_1, contours_2)
    contours_3 = remove_neighboring_pixels(contours_1, contours_3)

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
    path_to_images,
    path_to_segmentations,
    all_files,
    views,
    dicom_properties,
    segmentation_properties,
    colors_for_labels,
    font_size=8,
    dpi_value=100,
    length_ext=7
):
    """Function to plot area-time curves for all patients.

    Args:
        path_to_images (str): Directory of the folder with the echo images.
        path_to_segmentations (str): Directory of the folder with the segmentations.
        all_files (list): List of all files in the directory.
        views (list): List of views of the segmentations.
        dicom_properties (dict): Dictionary with dicom properties of all patients.
        segmentation_properties (dict): Dictionary with segmentation properties of all patients.
        colors_for_labels (np.ndarray): Color definitions for each label.
        font_size (int): Font size of the figure (default: 8).
        dpi_value (int): DPI value of the figure (default: 100).
        length_ext (int): Length of the file extension (default: 7 (.nii.gz)).
    """
    # Define directories of images and segmentations.
    for view in views:
        # Get dicom and segmentation properties of one patient
        ed_points = segmentation_properties["ED Points"][view]
        es_points = segmentation_properties["ES Points"][view]
        lv_areas = segmentation_properties["LV areas"][view]
        myo_areas = segmentation_properties["MYO areas"][view]
        la_areas = segmentation_properties["LA areas"][view]
        frame_times = dicom_properties["Times Frames"][view]

        # Plotting settings
        min_y_val = int(0.9 * min(min(lv_areas), min(myo_areas), min(la_areas)))
        max_y_val = int(1.1 * max(max(lv_areas), max(myo_areas), max(la_areas)))

        # Get all files of one view of one person.
        files_of_view = get_list_with_files_of_view(all_files, view)

        # Plot area-time curves for all frames of one patient. 
        # Only plot the first frame for now.
        for frame_nr, file_name in enumerate(files_of_view[0:1]):
            # Define file location and load echo image frame. 
            file_location_image = os.path.join(
                path_to_images, (file_name[:-length_ext] + "_0000" + file_name[-length_ext:])
            )
            echo_image = get_image_array(file_location_image)

            # Define file location and load segmentation. 
            file_location_seg = os.path.join(path_to_segmentations, file_name)
            seg = get_image_array(file_location_seg)
            seg_colored = color_segmentation(seg, colors_for_labels)

            contours_seg = color_contours_segmentation(
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
                        frame_times, lv_areas, color="green", label="Left ventricle"
                    )
                    plt.plot(frame_times, myo_areas, color="red", label="Myocardium")
                    plt.plot(frame_times, la_areas, color="blue", label="Left atrium")

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
