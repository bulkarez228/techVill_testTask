import cv2
import numpy as np
import open3d as o3d
import numpy as np
from geometry_utils import T_to_TUM_format, rvec_tvec_to_T

def estimate_aruco_pose(image, camera_matrix, dist_coeffs, marker_size=0.09):
    """
    Детекция маркеров и оценка их поз.
    """
    # Для старых версий OpenCV используется cv2.aruco, в новых — класс ArucoDetector
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    parameters = cv2.aruco.DetectorParameters()

    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
    corners, ids, rejected = detector.detectMarkers(image)

    poses = {}
    if ids is not None:
        for i, marker_id in enumerate(ids.flatten()):
            # corners[i] имеет форму (1, 4, 2)
            # Оценка позы одного маркера
            obj_points = np.array([
                [-marker_size / 2, marker_size / 2, 0],
                [marker_size / 2, marker_size / 2, 0],
                [marker_size / 2, -marker_size / 2, 0],
                [-marker_size / 2, -marker_size / 2, 0]
            ], dtype=np.float32)

            success, rvec, tvec = cv2.solvePnP(
                obj_points, corners[i][0], camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
            )
            if success:
                poses[int(marker_id)] = (rvec, tvec)

    return poses


def run_open3d_rgbd_odometry(sync_data, K_small):
    """
    Оптимизированная по скорости RGB-D одометрия Open3D.
    Работает значительно быстрее за счет уменьшения разрешения и ограничения итераций.
    """
    trajectory_TUM = []

    if len(sync_data) < 2:
        print("Недостаточно кадров для расчета одометрии.")
        return trajectory_TUM

    # 1. Задаем уменьшенное (в 2 раза) разрешение относительно оригинального кадра глубины
    # Оригинал глубины: 848x530 -> Цель: 424x265 (в 4 раза меньше пикселей)
    target_w = 848
    target_h = 530

    orig_h, orig_w = sync_data[0]['small_rgb'].shape[:2]

    # Считаем масштаб калибровки относительно оригинального цветного кадра (1280x800)
    scale_x = target_w / orig_w
    scale_y = target_h / orig_h

    # Масштабируем K_small под разрешение 424x265
    K_small_scaled = K_small.copy()
    K_small_scaled[0, 0] *= scale_x  # fx
    K_small_scaled[1, 1] *= scale_y  # fy
    K_small_scaled[0, 2] *= scale_x  # cx
    K_small_scaled[1, 2] *= scale_y  # cy

    # Инициализируем интринсики Open3D
    intrinsic_o3d = o3d.camera.PinholeCameraIntrinsic()
    intrinsic_o3d.set_intrinsics(
        target_w, target_h,
        K_small_scaled[0, 0], K_small_scaled[1, 1],
        K_small_scaled[0, 2], K_small_scaled[1, 2]
    )

    # 2. Настраиваем быстрые параметры оптимизатора
    option = o3d.pipelines.odometry.OdometryOption()
    # Ограничиваем количество итераций на каждом уровне пирамиды (уровни: грубый -> точный)
    # По умолчанию итераций значительно больше, уменьшение до [10, 5, 2] дает огромный прирост скорости
    option.iteration_number_per_pyramid_level = o3d.utility.IntVector([30, 20, 10])
    option.depth_diff_max = 0.03  # максимальная разница глубин для сопоставления точек (3 см)

    jacobian = o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm()

    # Стартовая поза камеры
    T_world_cam = np.eye(4)
    trajectory_TUM.append(T_to_TUM_format(T_world_cam, sync_data[0]['time']))

    last_successful_trans = np.eye(4)

    print(f"Запуск быстрой RGB-D одометрии (разрешение: {target_w}x{target_h})...")

    for i in range(len(sync_data) - 1):
        frame_src = sync_data[i]
        frame_dst = sync_data[i + 1]

        # 3. Ресайзим цветные кадры (билинейно)
        rgb_src_resized = cv2.resize(frame_src['small_rgb'], (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        rgb_dst_resized = cv2.resize(frame_dst['small_rgb'], (target_w, target_h), interpolation=cv2.INTER_LINEAR)

        # 4. Ресайзим кадры глубины (СТРОГО ближайшим соседом INTER_NEAREST!)
        depth_src_resized = cv2.resize(frame_src['small_depth'], (target_w, target_h), interpolation=cv2.INTER_NEAREST)
        depth_dst_resized = cv2.resize(frame_dst['small_depth'], (target_w, target_h), interpolation=cv2.INTER_NEAREST)

        # Конвертируем цвета BGR -> RGB
        rgb_src_rgb = cv2.cvtColor(rgb_src_resized, cv2.COLOR_BGR2RGB)
        rgb_dst_rgb = cv2.cvtColor(rgb_dst_resized, cv2.COLOR_BGR2RGB)

        # Оборачиваем в структуры Open3D
        color_src = o3d.geometry.Image(rgb_src_rgb)
        depth_src = o3d.geometry.Image(depth_src_resized)

        color_dst = o3d.geometry.Image(rgb_dst_rgb)
        depth_dst = o3d.geometry.Image(depth_dst_resized)

        # Создаем RGB-D образы
        rgbd_src = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_src, depth_src, depth_scale=1000.0, depth_trunc=3.0, convert_rgb_to_intensity=True
        )
        rgbd_dst = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_dst, depth_dst, depth_scale=1000.0, depth_trunc=3.0, convert_rgb_to_intensity=True
        )

        # Оценка смещения
        odo_init = np.eye(4)
        success, T_dst_src, _ = o3d.pipelines.odometry.compute_rgbd_odometry(
            rgbd_src, rgbd_dst, intrinsic_o3d, odo_init, jacobian, option
        )

        if success:
            step_transform = np.linalg.inv(T_dst_src)
            last_successful_trans = step_transform
        else:
            step_transform = last_successful_trans

        T_world_cam = T_world_cam @ step_transform
        trajectory_TUM.append(T_to_TUM_format(T_world_cam, frame_dst['time']))

        # Периодически выводим прогресс
        if (i + 1) % 500 == 0 or (i + 1) == len(sync_data) - 1:
            print(f"Обработано кадров: {i + 1}/{len(sync_data) - 1}")

    return trajectory_TUM


# Добавить в src/tracker.py (или в исполняемый файл)
def run_inside_out_localization(sync_data, board_corners_3d, K_small, dist_small):
    """
    Рассчитывает траекторию малой камеры по прямому наблюдению доски.
    """
    trajectory_TUM = []

    # Будем использовать детектор ArUco
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

    for idx, frame in enumerate(sync_data):
        timestamp = frame['time']
        img = frame['small_rgb']

        # Детектируем маркеры на кадре малой камеры
        corners, ids, _ = detector.detectMarkers(img)

        if ids is not None:
            ids_flat = ids.flatten()

            # Собираем пары соответствий 3D-2D для всех видимых маркеров доски (20, 21, 23)
            object_points = []
            image_points = []

            for i, m_id in enumerate(ids_flat):
                if m_id in [20, 21, 23] and m_id in board_corners_3d:
                    # 3D координаты углов на доске
                    object_points.append(board_corners_3d[m_id])
                    # 2D координаты углов на изображении (формат corners[i] - 1x4x2)
                    image_points.append(corners[i].reshape(4, 2))

            # Если виден хотя бы один маркер (минимум 4 соответствия точек для PnP)
            if len(object_points) > 0:
                obj_pts = np.vstack(object_points).astype(np.float32)
                img_pts = np.vstack(image_points).astype(np.float32)

                # Решаем задачу PnP для поиска позы доски относительно малой камеры (T_S_B)
                success, rvec, tvec = cv2.solvePnP(
                    obj_pts, img_pts, K_small, dist_small, flags=cv2.SOLVEPNP_SQPNP
                )

                if success:
                    T_S_B = rvec_tvec_to_T(rvec, tvec)
                    # Нам нужна поза малой камеры относительно доски: T_B_S = (T_S_B)^-1
                    T_B_S = np.linalg.inv(T_S_B)

                    # Сохраняем в TUM формате
                    tum_line = T_to_TUM_format(T_B_S, timestamp)
                    trajectory_TUM.append(tum_line)

    return trajectory_TUM