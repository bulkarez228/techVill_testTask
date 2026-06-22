# align_trajectory.py
import numpy as np
import os
from scipy.spatial.transform import Rotation as R
from geometry_utils import T_to_TUM_format


def load_tum_trajectory(filepath):
    """Загружает TUM файл в словарь {timestamp: T_4x4}"""
    poses = {}
    if not os.path.exists(filepath):
        return poses
    with open(filepath, 'r') as f:
        for line in f:
            if line.strip().startswith("#") or not line.strip():
                continue
            parts = list(map(float, line.strip().split()))
            t = parts[0]
            pos = np.array(parts[1:4])
            quat = np.array(parts[4:8])  # [qx, qy, qz, qw]

            T = np.eye(4)
            T[:3, :3] = R.from_quat(quat).as_matrix()
            T[:3, 3] = pos
            poses[t] = T
    return poses


# Добавить/заменить в align_trajectory.py

def find_umeyama_alignment(source_pts, target_pts, estimate_scale=True):
    """
    Алгоритм Умеямы (Sim3) для нахождения R, t, s, минимизирующих ||s * R * X + t - Y||_F.
    source_pts: (N, 3) - точки одометрии
    target_pts: (N, 3) - точки эталона
    """
    N, m = source_pts.shape
    mu_s = source_pts.mean(axis=0)
    mu_t = target_pts.mean(axis=0)

    ds = source_pts - mu_s
    dt = target_pts - mu_t

    sigma_s = (ds * ds).sum(axis=1).mean()
    cov = (dt.T @ ds) / N

    U, d, Vt = np.linalg.svd(cov)
    S = np.eye(m)

    if np.linalg.det(cov) < 0:
        S[m - 1, m - 1] = -1

    R_mat = U @ S @ Vt

    if estimate_scale:
        # Считаем коэффициент масштаба s
        scale = np.sum(d * S.diagonal()) / sigma_s
    else:
        scale = 1.0

    t_vec = mu_t - scale * R_mat @ mu_s

    # Собираем матрицу Sim3 перехода
    T = np.eye(4)
    T[:3, :3] = scale * R_mat
    T[:3, 3] = t_vec
    return T, scale

def find_kabsch_alignment(source_pts, target_pts):
    """
    Алгоритм Кабша для нахождения жесткого преобразования (R, t)
    source_pts: (N, 3) - точки одометрии
    target_pts: (N, 3) - точки эталона
    """
    centroid_src = np.mean(source_pts, axis=0)
    centroid_tgt = np.mean(target_pts, axis=0)

    ys = source_pts - centroid_src
    yt = target_pts - centroid_tgt

    # Вычисляем матрицу ковариации
    H = ys.T @ yt
    U, S, Vt = np.linalg.svd(H)

    R_mat = Vt.T @ U.T

    # Проверка на зеркальное отражение
    if np.linalg.det(R_mat) < 0:
        Vt[2, :] *= -1
        R_mat = Vt.T @ U.T

    t_vec = centroid_tgt - R_mat @ centroid_src

    T = np.eye(4)
    T[:3, :3] = R_mat
    T[:3, 3] = t_vec
    return T


def align_and_save_odometry(odo_path, ref_path, output_path):
    odo_poses = load_tum_trajectory(odo_path)
    ref_poses = load_tum_trajectory(ref_path)

    if not odo_poses or not ref_poses:
        print("Ошибка: Один из файлов пуст или не найден.")
        return

    common_times = sorted(list(set(odo_poses.keys()).intersection(set(ref_poses.keys()))))

    if len(common_times) < 10:
        print("Слишком мало перекрытий по времени для надежного выравнивания.")
        return

    src_pts = np.array([odo_poses[t][:3, 3] for t in common_times])
    tgt_pts = np.array([ref_poses[t][:3, 3] for t in common_times])

    # Вычисляем матрицу перехода T_B_C с учетом масштаба через Умеяму
    T_B_C, scale = find_umeyama_alignment(src_pts, tgt_pts, estimate_scale=False)
    print(f"Вычислено Sim3 выравнивание Умеямы:")
    print(f"-> Масштабный коэффициент (scale): {scale:.4f}")
    print(f"-> Сдвиг (трансляция): {T_B_C[:3, 3]} метров")

    # Применяем выравнивание
    aligned_poses = {}
    for t, T_odo in odo_poses.items():
        # Для корректного преобразования 6DoF позы с масштабом:
        # Вращательную часть умножаем на масштаб, трансляцию переносим,
        # но сохраняем ортонормированность вращения камеры (шкалу масштаба применяем к позиции)
        T_aligned = np.eye(4)
        # R_aligned = R_match * R_odo
        T_aligned[:3, :3] = (T_B_C[:3, :3] / scale) @ T_odo[:3, :3]
        # t_aligned = s * R_match * t_odo + t_match
        T_aligned[:3, 3] = T_B_C[:3, :3] @ T_odo[:3, 3] + T_B_C[:3, 3]

        aligned_poses[t] = T_aligned

    with open(output_path, 'w') as f:
        f.write("# Aligned Camera Odometry (Open3D Umeyama Sim3)\n")
        for t in sorted(aligned_poses.keys()):
            f.write(T_to_TUM_format(aligned_poses[t], t) + "\n")

    print(f"Выровненная одометрия сохранена в {output_path}.")

