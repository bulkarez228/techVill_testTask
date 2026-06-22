import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

def rvec_tvec_to_T(rvec, tvec):
    """
    Преобразует вектор поворота (Rodrigues) и вектор трансляции в матрицу 4x4.
    """
    R_mat, _ = cv2.Rodrigues(rvec)
    T = np.eye(4)
    T[:3, :3] = R_mat
    T[:3, 3] = tvec.flatten()
    return T

def T_to_TUM_format(T, timestamp):
    """
    Преобразует матрицу 4x4 в строку формата TUM:
    timestamp x y z qx qy qz qw
    """
    t = T[:3, 3]
    rot = R.from_matrix(T[:3, :3])
    quat = rot.as_quat()  # Возвращает [qx, qy, qz, qw]
    return f"{timestamp:.6f} {t[0]:.6f} {t[1]:.6f} {t[2]:.6f} {quat[0]:.6f} {quat[1]:.6f} {quat[2]:.6f} {quat[3]:.6f}"


def calibrate_board_corners_3d(T_L_B, board_poses_in_L, marker_size=0.09):
    """
    Вычисляет точные 3D координаты углов маркеров доски (20, 21, 23)
    в единой системе координат доски B.

    board_poses_in_L: словарь {marker_id: (rvec, tvec)} для маркеров стола.
    """
    s = marker_size
    # Локальные 3D координаты 4 углов одиночного маркера (по часовой стрелке)
    local_corners = np.array([
        [-s / 2, s / 2, 0],
        [s / 2, s / 2, 0],
        [s / 2, -s / 2, 0],
        [-s / 2, -s / 2, 0]
    ], dtype=np.float32)

    T_B_L = np.linalg.inv(T_L_B)
    board_corners_3d = {}

    for m_id in [20, 21, 23]:
        if m_id in board_poses_in_L:
            rvec, tvec = board_poses_in_L[m_id]
            T_L_m = rvec_tvec_to_T(rvec, tvec)

            # Находим позу маркера относительно начала системы координат доски B
            T_B_m = T_B_L @ T_L_m

            # Переводим все 4 угла маркера в систему координат доски
            corners_3d_B = []
            for corner in local_corners:
                pt_homogeneous = np.append(corner, 1.0)
                pt_B = T_B_m @ pt_homogeneous
                corners_3d_B.append(pt_B[:3])

            board_corners_3d[m_id] = np.array(corners_3d_B, dtype=np.float32)

    return board_corners_3d