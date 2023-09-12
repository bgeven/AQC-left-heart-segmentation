import os
import numpy as np
from scipy.interpolate import interp1d


def correct_for_zeros(timings, areas):
    """Correct for zero values in the area list by interpolating between the non-zero values.

    Args:
        timings (list): List of timings of frames.
        areas (list): List of areas over time for a specific structure.

    Returns:
        areas_adapted (list): List of areas with zero values interpolated between the non-zero values.
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
        areas_adapted = f(timings)

    else:
        areas_adapted = areas

    return areas_adapted


def normalise_list(list_to_normalise):
    """Normalise a list of values to a range of 0 to 1.

    Args:
        list_to_normalise (list): List of values to normalise.

    Returns:
        normalised_list (list): Normalised list of values.
    """
    # Find the minimum and maximum values in the list.
    min_value = min(list_to_normalise)
    max_value = max(list_to_normalise)

    # Find the range of values in the list.
    value_range = max_value - min_value

    # Normalise the list of values.
    normalised_list = [(value - min_value) / value_range for value in list_to_normalise]

    return normalised_list


def get_part(values, frame_timings, time_points, nr_time_points, normalise="no"):

    values_part = values[time_points[0] : time_points[1] + 1]
    timings_part = frame_timings[time_points[0] : time_points[1] + 1]

    timings_adj = [item - timings_part[0] for item in timings_part]
    total_time = timings_adj[-1] - timings_adj[0]

    values_adj = correct_for_zeros(timings_part, values_part)

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
