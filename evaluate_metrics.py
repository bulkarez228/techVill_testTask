# evaluate_metrics.py
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from pseudo_gt import load_tum_trajectory, calibrate_head_to_sensor, find_overlapping_frames


def compute_trajectory_errors(est_poses, gt_poses):
    """
    Вычисляет ошибки ATE для трансляции (в мм) и вращения (в градусах)
    для пересекающихся по времени кадров.
    """
    common_times = sorted(list(set(est_poses.keys()).intersection(set(gt_poses.keys()))))

    times = []
    trans_errors = []
    rot_errors = []

    for t in common_times:
        T_est = est_poses[t]
        T_gt = gt_poses[t]

        # 1. Ошибка трансляции (в миллиметрах)
        t_err = np.linalg.norm(T_est[:3, 3] - T_gt[:3, 3]) * 1000.0
        trans_errors.append(t_err)

        # 2. Геодезическая ошибка вращения на SO(3) (в градусах)
        R_err = np.linalg.inv(T_est[:3, :3]) @ T_gt[:3, :3]
        trace = np.trace(R_err)
        # Стабилизация arccos от численных ошибок округления
        val = np.clip((trace - 1.0) / 2.0, -1.0, 1.0)
        r_err = np.arccos(val) * (180.0 / np.pi)
        rot_errors.append(r_err)

        times.append(t)

    return np.array(times), np.array(trans_errors), np.array(rot_errors)


def print_metrics_table(method_name, trans_err, rot_err):
    """Выводит статистику ошибок метода"""
    if len(trans_err) == 0:
        print(f"| {method_name:<30} | Нет пересечений по времени |")
        return

    t_mean = np.mean(trans_err)
    t_median = np.median(trans_err)
    t_p95 = np.percentile(trans_err, 95)
    t_rmse = np.sqrt(np.mean(np.square(trans_err)))

    r_mean = np.mean(rot_err)
    r_median = np.median(rot_err)
    r_p95 = np.percentile(rot_err, 95)
    r_rmse = np.sqrt(np.mean(np.square(rot_err)))

    print(
        f"| {method_name:<30} | {t_mean:8.2f} | {t_median:8.2f} | {t_p95:8.2f} | {t_rmse:8.2f} | {r_mean:8.2f} | {r_median:8.2f} | {r_p95:8.2f} | {r_rmse:8.2f} |")

