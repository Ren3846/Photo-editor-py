import numpy as np

NEIGHBOUR_COUNT = 5
KNN_WINDOW = 1


def apply_filter(brightness_matrix: np.ndarray, row_index: int, column_index: int) -> int:
    row_index_low = row_index - 1 if row_index > 0 else 0
    row_index_high = row_index + 2 if row_index < brightness_matrix.shape[0] - 2 else brightness_matrix.shape[0] - 1

    column_index_low = column_index - 1 if column_index > 0 else 0
    column_index_high = column_index + 2 if column_index < brightness_matrix.shape[1] - 2 else brightness_matrix.shape[
                                                                                                   1] - 1

    values_in_window = brightness_matrix[row_index_low:row_index_high, column_index_low:column_index_high].flatten()
    values_in_window_sorted = np.sort(values_in_window)

    mid_neighbour = (len(values_in_window_sorted) + 1) // 2
    lowest_valuable_neighbour = (mid_neighbour - NEIGHBOUR_COUNT // 2) \
        if (mid_neighbour - NEIGHBOUR_COUNT // 2) > 0 \
        else 0
    highest_valuable_neighbour = mid_neighbour + NEIGHBOUR_COUNT // 2  \
        if (mid_neighbour + NEIGHBOUR_COUNT // 2) < len(values_in_window) \
        else len(values_in_window) - 1

    valuable_values = values_in_window_sorted[lowest_valuable_neighbour:highest_valuable_neighbour + 1]

    return np.mean(valuable_values, dtype=np.float64)


def knn_filter(brightness_matrix: np.ndarray):
    filtered_image = np.zeros(shape=brightness_matrix.shape)
    for i in range(brightness_matrix.shape[0]):
        for j in range(brightness_matrix.shape[1]):
            filtered_image[i, j] = apply_filter(brightness_matrix, i, j)

    return filtered_image
