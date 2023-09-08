# Functions to create figures. 
import os
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from general_utilities import get_image_array, separate_segmentation, find_contours

def color_segmentation(seg, colors_for_labels):
    """Function to color a segmentation with colors for each label.

    Args:
        seg (numpy array): Segmentation to be colored.
        colors_for_labels (numpy array): Color definitions for each label.
    
    Returns:
        seg_colored (numpy array): Colored segmentation.
    """
    seg_colored = colors_for_labels[seg]

    return seg_colored

def remove_neighboring_pixels(contours1, contours2):  
    """Function to remove points on the contours that neighbor the LV contour.

    Args:
        contours1 (list): List of contours of the LV.
        contours2 (list): List of contours of the MYO or LA.

    Returns:
        contour_adapted (list): List of contours of the MYO or LA with points removed.
    """  
    if len(contours1) > 0 and len(contours2) > 0:
        contour_adapted = []

        # Loop over all points on the contours of the MYO or LA to check if they neighbor the LV contour.
        for contour2 in contours2:
            for j in range(len(contour2)):
                x_coor2 = int(contour2[j][0][0])
                y_coor2 = int(contour2[j][0][1])
                
                coor2 = tuple([x_coor2, y_coor2])
                
                min_distance = cv2.pointPolygonTest(contours1[0], coor2, True)
                
                # If the distance between the point on the MYO or LA contour and the LV contour is larger than 3 pixels, add the point to the list of points.
                if abs(min_distance) > 3.0:
                    contour_adapted.append([x_coor2, y_coor2])
        
        # Convert list of points to a numpy array and reshape to the correct format.
        contour_adapted = np.array(contour_adapted)
        contour_adapted = tuple(contour_adapted.reshape((contour_adapted.shape[0], 1, 2))[np.newaxis, :])
        
    else:
        contour_adapted = contours2
    
    return contour_adapted

def color_contours_segmentation(image, seg, label_colors):
    """Function to project contours of a segmentation on an image.

    Args:
        image (numpy array): Image to project contours on.
        seg (numpy array): Segmentation to project contours of.
        label_colors (numpy array): Color definitions for each label.

    Returns:
        image_with_contours (numpy array): Image with projected contours.
    """
    image_with_contours = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    
    # Map each label value to a specific color
    colored_labels = color_segmentation(seg, label_colors)

    # Convert the colored labels back to an image
    image_for_blend = Image.fromarray(image_with_contours) #.convert('RGB')
    color_labels = Image.fromarray(colored_labels.astype('uint8'))
    
    # Create blended image
    overlay_image = Image.blend(image_for_blend, color_labels, 0.1)
    image_with_contours = np.array(overlay_image)
    
    # Separate contours for each label
    _, seg1, seg2, seg3 = separate_segmentation(seg)
    
    contours1 = find_contours(seg1, 'all')
    contours2 = find_contours(seg2, 'all')
    contours3 = find_contours(seg3, 'all')
    
    # Remove points on the contours that neighbor the LV contour
    contours2 = remove_neighboring_pixels(contours1, contours2)
    contours3 = remove_neighboring_pixels(contours1, contours3)
    
    # Draw all contours on the image
    for contour2 in contours2:
        cv2.drawContours(image_with_contours, contour2, -1, (int(label_colors[2][0]), int(label_colors[2][1]), int(label_colors[2][2])), 4)
      
    for contour3 in contours3:
        cv2.drawContours(image_with_contours, contour3, -1, (int(label_colors[3][0]), int(label_colors[3][1]), int(label_colors[3][2])), 4)
    
    cv2.drawContours(image_with_contours, contours1, -1, (int(label_colors[1][0]), int(label_colors[1][1]), int(label_colors[1][2])), 4)   
    
    return image_with_contours

def main_plot_area_time_curves(directory_images, directory_segmentations, dicom_properties, segmentation_properties, colors_for_labels):
    """Function to plot area-time curves for all patients.

    Args:
        directory_images (string): Directory of the folder with the US images.
        directory_segmentations (string): Directory of the folder with the segmentations.
        dicom_properties (dictionary): Dictionary with dicom properties of all patients.
        segmentation_properties (dictionary): Dictionary with segmentation properties of all patients.
        colors_for_labels (numpy array): Color definitions for each label.
    """
    # Get list of filenames in one folder
    all_files = os.listdir(directory_segmentations)
    patients = sorted(set([i[:29] for i in all_files if i.startswith('cardiohance')]))

    # Loop over all files in a folder
    for patient in patients:
        # Get dicom and segmentation properties of one patient
        ED_points = segmentation_properties['ED Points'][patient]
        ES_points = segmentation_properties['ES Points'][patient]
        LV_areas = segmentation_properties['LV areas'][patient]
        MYO_areas = segmentation_properties['MYO areas'][patient]
        LA_areas = segmentation_properties['LA areas'][patient]
        frame_times = dicom_properties['Times Frames'][patient]
        
        # Plotting settings
        min_y_val = int(0.9 * min(min(LV_areas), min(MYO_areas), min(LA_areas)))
        max_y_val = int(1.1 * max(max(LV_areas), max(MYO_areas), max(LA_areas)))
        
        # Find filenames for all images of one patient and sort based on frame number
        images_of_one_person_unsorted = [i for i in all_files if i.startswith(patient)]    
        images_of_one_person = sorted(images_of_one_person_unsorted, key=lambda x: int(x[30:-7]))
        
        # Loop over all images of one person
        for frame_nr, image in enumerate(images_of_one_person[0:1]):
            # Define file location and load US image frame
            file_location_image = os.path.join(directory_images, (image[:33] + '_0000' + image[-7:]))
            US_image = get_image_array(file_location_image)
            
            # Define file location and load segmentation
            file_location_seg = os.path.join(directory_segmentations,image)
            seg_uncolored = get_image_array(file_location_seg)          
            seg = color_segmentation(seg_uncolored, colors_for_labels)
                       
            contours_seg = color_contours_segmentation(US_image, seg_uncolored, colors_for_labels)
                           
            plt.figure(dpi=300)
            plt.suptitle(('Segmentation of ' + patient + ', frame ' + str(frame_nr)))
            
            # Format figure with subplots
            X = [ (2,3,1), (2,3,2), (2,3,3), (2,3,(4,5)) ]
            font_size = 8
            
            for nr_plot, (nrows, ncols, plot_number) in enumerate(X):
                if nr_plot == 0:
                    plt.subplot(nrows, ncols, plot_number)
                    plt.imshow(US_image, cmap='gray')
                    plt.title('Echo image', fontsize=font_size, loc='left')
                    plt.axis('off')
                    
                elif nr_plot == 1:
                    plt.subplot(nrows, ncols, plot_number)
                    plt.imshow(seg)
                    plt.title('Segmentation', fontsize=font_size, loc='left')
                    plt.axis('off')
                    
                elif nr_plot == 2:
                    plt.subplot(nrows, ncols, plot_number)
                    plt.imshow(contours_seg, cmap='gray', interpolation='none')
                    plt.title('Overlay', fontsize=font_size, loc='left')
                    plt.axis('off')
                    
                elif nr_plot == 3:
                    plt.subplot(nrows, ncols, plot_number)
                    plt.title('Area-time curves', fontsize=font_size, loc='left')
                    plt.plot(frame_times, LV_areas, color='green', label='Left ventricle')
                    plt.plot(frame_times, MYO_areas, color='red', label='Myocardium')
                    plt.plot(frame_times, LA_areas, color='blue', label='Left atrium')

                    # Plot vertical lines for current frame, ES and ED points
                    plt.axvline(x = frame_times[frame_nr], color = 'c', linewidth=1, linestyle='--', label='Time of frame')

                    # Plot vertical lines for all ED and ES points 
                    plt.axvline(x = frame_times[ES_points[0]], color = 'm', linewidth=1, linestyle='--', label='ES point')
                    plt.axvline(x = frame_times[ED_points[0]], color = 'y', linewidth=1, linestyle='--', label='ED point')
                    for idx in range(1, len(ES_points)):
                        plt.axvline(x = frame_times[ES_points[idx]], color = 'm', linewidth=1, linestyle='--')
                    for idx in range(1, len(ED_points)):
                        plt.axvline(x = frame_times[ED_points[idx]], color = 'y', linewidth=1, linestyle='--')

                    plt.xlabel('Time [ms]')
                    plt.ylabel('Area [cm$^{2}$]')
                    plt.xlim(frame_times[0]-40, frame_times[-1]+40)
                    plt.ylim(min_y_val, max_y_val)
                    plt.legend(bbox_to_anchor=(1, 0.5), loc="center left")