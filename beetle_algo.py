import numpy as np
from typing import Tuple, List

WHITE_PIXEL = 1
BLACK_PIXEL = 0
EXTERNAL_STOP_LIMIT = 1000


def get_start_point(black_white_matrix: np.ndarray) -> np.ndarray:
    return np.where(black_white_matrix == WHITE_PIXEL)


def turn_left(movement_vector: np.ndarray) -> np.ndarray:
    matrix = np.array([
        [0, -1],
        [1, 0]
    ])

    return np.matmul(matrix, movement_vector.T)


def turn_right(movement_vector: np.ndarray) -> np.ndarray:
    matrix = np.array([
        [0, 1],
        [-1, 0]
    ])

    return np.matmul(matrix, movement_vector)


def get_image_segments(black_white_matrix: np.ndarray, segments_count: int = 3) -> List[List[Tuple[int, int]]]:
    black_white_matrix_2d = black_white_matrix[:, :, 0]
    # normalized from 0 to 1
    black_white_matrix_normalized = (black_white_matrix_2d - np.min(black_white_matrix_2d)) / np.ptp(
        black_white_matrix_2d)

    coordinates_array = np.column_stack(get_start_point(black_white_matrix_normalized))

    segments = []
    for i in range(segments_count):
        start_point = np.array([coordinates_array[0, 0], coordinates_array[0, 1]])
        segment = beetle(black_white_matrix_normalized, start_point)

        segment_min_row = min(map(lambda coordinates: coordinates[0], segment))
        segment_max_row = max(map(lambda coordinates: coordinates[0], segment))
        segment_min_column = min(map(lambda coordinates: coordinates[1], segment))
        segment_max_column = max(map(lambda coordinates: coordinates[1], segment))

        coordinates_array = coordinates_array[
            (
                    (coordinates_array[:, 0] < segment_min_row)
                    | (coordinates_array[:, 0] > segment_max_row)
                    | (coordinates_array[:, 1] < segment_min_column)
                    | (coordinates_array[:, 1] > segment_max_column)
            )
        ]

        segments.append(segment)

    return segments


def beetle(black_white_matrix: np.ndarray, start_point: np.ndarray) -> List[Tuple[int, int]]:
    path: List[Tuple[int, int]] = []
    movement_vector = np.array([0, 1])

    # 0 - вертикаль, 1 - горизонталь
    coordinates = start_point.copy()
    error_counter = 0
    while (True):
        if len(path) > 0 and path[0][0] == coordinates[0] and path[0][1] == coordinates[1]:
            break

        if error_counter == EXTERNAL_STOP_LIMIT:
            break

        # В случае если координаты жука невалидны - считаем что он на черной клетке
        if coordinates[0] < 0 \
                or coordinates[0] >= black_white_matrix.shape[0] \
                or coordinates[1] < 0 \
                or coordinates[1] >= black_white_matrix.shape[1]:
            movement_vector = turn_right(movement_vector)
            error_counter += 1
        # На белой клетке - поворачиваем налево
        elif black_white_matrix[coordinates[0], coordinates[1]] == WHITE_PIXEL:
            path.append((coordinates[0], coordinates[1]))
            movement_vector = turn_left(movement_vector)
            error_counter = 0
        # На черной клетке - поворачиваем направо
        else:
            movement_vector = turn_right(movement_vector)
            error_counter += 1

        coordinates += movement_vector

    return path
