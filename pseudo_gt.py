# pseudo_gt.py
import os
import numpy as np
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


def find_overlapping_frames(poses_a, poses_b, max_dt=0.01):
    """Находит пересекающиеся по времени кадры"""
    pairs = []
    times_b = np.array(sorted(poses_b.keys()))
    for t_a in sorted(poses_a.keys()):
        if len(times_b) == 0:
            continue
        idx = np.argmin(np.abs(times_b - t_a))
        t_b = times_b[idx]
        if np.abs(t_b - t_a) <= max_dt:
            pairs.append((t_a, t_b))
    return pairs


def calibrate_head_to_sensor(inside_out_poses, outside_in_poses):
    """
    Вычисляет жесткий калибровочный трансформ от маркера id1 к сенсору S.
    T_B_S = T_B_id1 * T_id1_S  =>  T_id1_S = (T_B_id1)^-1 * T_B_S
    """
    pairs = find_overlapping_frames(outside_in_poses, inside_out_poses)
    if not pairs:
        raise ValueError("Нет перекрывающихся по времени кадров для калибровки головы и сенсора!")

    transforms = []
    for t_out, t_in in pairs:
        T_B_id1 = outside_in_poses[t_out]
        T_B_S = inside_out_poses[t_in]
        transforms.append(np.linalg.inv(T_B_id1) @ T_B_S)

    # Находим устойчивое среднее (медиану для трансляции и среднее вращение)
    T_id1_S = np.eye(4)
    T_id1_S[:3, 3] = np.median([T[:3, 3] for T in transforms], axis=0)
    T_id1_S[:3, :3] = R.from_matrix([T[:3, :3] for T in transforms]).mean().as_matrix()
    return T_id1_S


def smooth_trajectory_sliding_window(trajectory, window_size=5):
    """
    Сглаживает траекторию скользящим средним на группе Ли SE(3).
    """
    timestamps = sorted(trajectory.keys())
    smoothed = {}
    half_w = window_size // 2

    for i, t in enumerate(timestamps):
        start_idx = max(0, i - half_w)
        end_idx = min(len(timestamps), i + half_w + 1)

        window_poses = [trajectory[timestamps[j]] for j in range(start_idx, end_idx)]

        # Усредняем трансляцию
        mean_t = np.mean([T[:3, 3] for T in window_poses], axis=0)
        # Усредняем вращение на группе Ли (через средний кватернион)
        mean_r = R.from_matrix([T[:3, :3] for T in window_poses]).mean().as_matrix()

        T_smooth = np.eye(4)
        T_smooth[:3, :3] = mean_r
        T_smooth[:3, 3] = mean_t
        smoothed[t] = T_smooth

    return smoothed


def evaluate_pseudo_gt_accuracy(inside_out_poses, outside_in_poses):
    """
    Оценивает погрешность эталона методом кросс-валидации (Leave-20%-Out)
    """
    pairs = find_overlapping_frames(outside_in_poses, inside_out_poses)
    n_pairs = len(pairs)
    if n_pairs < 10:
        return 0.0, 0.0

    # Делим на обучающую (80%) и валидационную (20%) выборки
    np.random.seed(42)
    indices = np.arange(n_pairs)
    np.random.shuffle(indices)
    split_idx = int(n_pairs * 0.8)

    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]

    # Считаем калибровку только на 80% данных
    train_transforms = []
    for idx in train_indices:
        t_out, t_in = pairs[idx]
        train_transforms.append(np.linalg.inv(outside_in_poses[t_out]) @ inside_out_poses[t_in])

    T_id1_S_train = np.eye(4)
    T_id1_S_train[:3, 3] = np.median([T[:3, 3] for T in train_transforms], axis=0)
    T_id1_S_train[:3, :3] = R.from_matrix([T[:3, :3] for T in train_transforms]).mean().as_matrix()

    # Проверяем на отложенных 20% данных
    translation_errors = []
    rotation_errors = []
    for idx in test_indices:
        t_out, t_in = pairs[idx]
        T_B_S_gt = inside_out_poses[t_in]
        T_B_S_pred = outside_in_poses[t_out] @ T_id1_S_train

        translation_errors.append(np.linalg.norm(T_B_S_gt[:3, 3] - T_B_S_pred[:3, 3]))
        r_err_mat = np.linalg.inv(T_B_S_pred[:3, :3]) @ T_B_S_gt[:3, :3]
        rotation_errors.append(np.linalg.norm(R.from_matrix(r_err_mat).as_euler('xyz', degrees=True)))

    rms_t = np.sqrt(np.mean(np.square(translation_errors)))
    rms_r = np.sqrt(np.mean(np.square(rotation_errors)))
    return rms_t, rms_r


def build_and_save_pseudo_gt(inside_out_path, outside_in_path, output_path):
    """Основная функция сборки Pseudo-GT"""
    inside_out_poses = load_tum_trajectory(inside_out_path)
    outside_in_poses = load_tum_trajectory(outside_in_path)

    if not inside_out_poses or not outside_in_poses:
        print("Ошибка: Входные файлы траекторий для Pseudo-GT пусты.")
        return

    # 1. Измеряем погрешность будущего эталона
    rms_t, rms_r = evaluate_pseudo_gt_accuracy(inside_out_poses, outside_in_poses)
    print(f"\n[Pseudo-GT Валидация] Оцененная погрешность эталона:")
    print(f"-> RMS Ошибка трансляции: {rms_t * 1000:.2f} мм")
    print(f"-> RMS Ошибка вращения: {rms_r:.3f} градусов")

    # 2. Калибруем жесткое смещение по всем перекрывающимся данным
    T_id1_S = calibrate_head_to_sensor(inside_out_poses, outside_in_poses)
    print(f"-> Вычислен вектор калибровки T_id1_S (длина): {np.linalg.norm(T_id1_S[:3, 3]) * 1000:.1f} мм")

    # 3. Фьюзинг (объединение потоков)
    pseudo_gt = {}
    all_times = sorted(list(set(list(inside_out_poses.keys()) + list(outside_in_poses.keys()))))

    for t in all_times:
        if t in inside_out_poses:
            pseudo_gt[t] = inside_out_poses[t]
        elif t in outside_in_poses:
            pseudo_gt[t] = outside_in_poses[t] @ T_id1_S

    # 4. Сглаживание скользящим окном на SE(3) для подавления дрожания маркеров
    smoothed_pseudo_gt = smooth_trajectory_sliding_window(pseudo_gt, window_size=5)

    # 5. Запись в файл
    with open(output_path, 'w') as f:
        f.write("# Pseudo-Ground-Truth Trajectory (Fused & Smoothed)\n")
        f.write(f"# Estimated Precision RMS Translation: {rms_t * 1000:.2f} mm\n")
        f.write(f"# Estimated Precision RMS Rotation: {rms_r:.3f} deg\n")
        for t in sorted(smoothed_pseudo_gt.keys()):
            f.write(T_to_TUM_format(smoothed_pseudo_gt[t], t) + "\n")

    print(f"-> Файл эталона успешно сохранен: {output_path}")