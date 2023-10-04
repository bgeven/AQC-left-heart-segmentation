# NOT FINISHED AT ALL!!!!
import os
from scipy.interpolate import interp1d
from functions.general_utilities import *


def interpolate_missing_areas(timings, areas):
    """Correct for missing values in the area list by interpolating between the non-zero values.

    Args:
        timings (list): List of timings of frames.
        areas (list): List of areas over time for a specific structure.

    Returns:
        areas_interpolated (list): List of areas over time for a specific structure, with missing values interpolated.
    """
    # If there are zero values in the list of areas, interpolate between the non-zero values.
    if min(areas) == 0:
        # Find the indices of the zero values.
        indices_zeros = [index for index, value in enumerate(areas) if value == 0]

        # Find the indices of the non-zero values.
        timings_no_zeros = [
            value for index, value in enumerate(timings) if index not in indices_zeros
        ]
        areas_no_zeros = [
            value for index, value in enumerate(areas) if index not in indices_zeros
        ]

        # Interpolate between the non-zero values.
        f = interp1d(timings_no_zeros, areas_no_zeros, kind="linear")
        areas_interpolated = f(timings)

    else:
        areas_interpolated = areas

    return areas_interpolated

def get_part(values, frame_timings, time_points, nr_time_points, normalise="no"):

    values_part = values[time_points[0] : time_points[1] + 1]
    timings_part = frame_timings[time_points[0] : time_points[1] + 1]

    timings_adj = [item - timings_part[0] for item in timings_part]
    total_time = timings_adj[-1] - timings_adj[0]

    values_adj = interpolate_missing_areas(timings_part, values_part)

    timings_int = list(np.linspace(0, total_time, nr_time_points, endpoint=True))

    f = interp1d(timings_adj, values_adj, kind="linear")
    values_return = f(timings_int)

    if normalise == "yes":
        values_return = normalise_list(values_return)

    # plt.figure()
    # plt.plot(timings_adj, values_part)
    # plt.plot(timings_int, values_int)

    # plt.figure(dpi=300)
    # plt.subplot(121)
    # plt.plot(timings_int, values_int)
    # plt.subplot(122)
    # plt.plot(timings_int, values_normalized)

    return values_return, timings_int


def comp_sys_dia_cycle_ratio(path_to_dataset, patients, dicom_properties, segmentation_properties, cycle_information):
    perc_LV_sys_list, perc_LV_dia_list, perc_LA_sys_list, perc_LA_dia_list = [], [], [], []
    for patient in patients:
        all_files = os.listdir(os.path.join(path_to_dataset, patient))
        views = get_list_with_views(all_files)

        for view in views:      
            ED_points_cycle = cycle_information['ED_points_selected'][view]
            ES_point_cycle = cycle_information['ES_point_selected'][view]   
            
            LV_areas_all = segmentation_properties['LV areas'][view]
            LA_areas_all = segmentation_properties['LA areas'][view]
            
            frame_timings_all = dicom_properties['Frame Times'][view]
            timings_total = frame_timings_all[ED_points_cycle[0]:ED_points_cycle[1]+1]
            
            # LV
            timings_LV_systolic = frame_timings_all[ED_points_cycle[0]:ES_point_cycle[0]+1]    
            tot_time_LV_systolic = timings_LV_systolic[-1] - timings_LV_systolic[0]
            
            # LA
            # Get different ES point for LA areas
            LA_selection = LA_areas_all[ED_points_cycle[0]:ED_points_cycle[1]+1]
            max_LA = max(LA_selection)
            ES_point_LA = LA_selection.index(max_LA) + ED_points_cycle[0]
            
            timings_LA_systolic = frame_timings_all[ED_points_cycle[0]:ES_point_LA+1]    
            tot_time_LA_systolic = timings_LA_systolic[-1] - timings_LA_systolic[0]
            
            tot_time = timings_total[-1] - timings_total[0]
            
            percentage_LV_systolic = tot_time_LV_systolic / tot_time
            percentage_LA_systolic = tot_time_LA_systolic / tot_time

            perc_LV_sys_list.append(percentage_LV_systolic)
            perc_LA_sys_list.append(percentage_LA_systolic)
        
    mean_perc_LV_sys = int(np.nanmean(perc_LV_sys_list)*100)
    mean_perc_LV_dia = 100 - mean_perc_LV_sys
    mean_perc_LA_sys = int(np.nanmean(perc_LA_sys_list)*100)
    mean_perc_LA_dia = 100 - mean_perc_LA_sys