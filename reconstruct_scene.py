# reconstruct_scene.py
import numpy as np
import cv2
import open3d as o3d
import os


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

            from scipy.spatial.transform import Rotation as R
            T = np.eye(4)
            T[:3, :3] = R.from_quat(quat).as_matrix()
            T[:3, 3] = pos
            poses[t] = T
    return poses


def reconstruct_tsdf(sync_data, trajectory, intrinsic, voxel_size=0.015, sdf_trunc=0.04):
    """
    Метод 1: Воксельная TSDF-интеграция.
    Если trajectory=None, используется единичная матрица (без использования траектории).
    """
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_size,
        sdf_trunc=sdf_trunc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
    )

    step = 3
    integrated_frames = 0

    for i in range(0, len(sync_data), step):
        frame = sync_data[i]
        t = frame['time']

        # Если траектория не задана, позиционируем камеру в центре (T = I)
        if trajectory is None:
            T_world_cam = np.eye(4)
        else:
            times = np.array(list(trajectory.keys()))
            if len(times) == 0:
                continue
            idx = np.argmin(np.abs(times - t))
            best_t = times[idx]

            # Порог сопоставления по времени 100 мс
            if np.abs(best_t - t) > 0.10:
                continue
            T_world_cam = trajectory[best_t]

        integrated_frames += 1
        target_h, target_w = frame['small_depth'].shape

        rgb_resized = cv2.resize(frame['small_rgb'], (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        rgb_rgb = cv2.cvtColor(rgb_resized, cv2.COLOR_BGR2RGB)

        color = o3d.geometry.Image(rgb_rgb)
        depth = o3d.geometry.Image(frame['small_depth'])

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color, depth, depth_scale=1000.0, depth_trunc=3.0, convert_rgb_to_intensity=False
        )

        extrinsic = np.linalg.inv(T_world_cam)
        volume.integrate(rgbd, intrinsic, extrinsic)

    print(f"  [TSDF] Успешно интегрировано кадров: {integrated_frames}")
    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    return mesh


def reconstruct_point_cloud_fusion(sync_data, trajectory, intrinsic, voxel_size=0.025):
    """
    Метод 2: Плотное облако точек с отсечением фона и глубокой фильтрацией шумов.
    """
    fused_pcd = o3d.geometry.PointCloud()

    step = 5
    integrated_frames = 0

    for i in range(0, len(sync_data), step):
        frame = sync_data[i]
        t = frame['time']

        if trajectory is None:
            T_world_cam = np.eye(4)
        else:
            times = np.array(list(trajectory.keys()))
            if len(times) == 0:
                continue
            idx = np.argmin(np.abs(times - t))
            best_t = times[idx]

            if np.abs(best_t - t) > 0.10:
                continue
            T_world_cam = trajectory[best_t]

        integrated_frames += 1
        target_h, target_w = frame['small_depth'].shape

        rgb_resized = cv2.resize(frame['small_rgb'], (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        rgb_rgb = cv2.cvtColor(rgb_resized, cv2.COLOR_BGR2RGB)

        color = o3d.geometry.Image(rgb_rgb)
        depth = o3d.geometry.Image(frame['small_depth'])

        # ВАЖНО: depth_trunc=1.3 отсекает все, что находится дальше 1.3 метров от камеры.
        # Это сотрет задний фон комнаты и оставит только стол и стабилизатор!
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color, depth, depth_scale=1000.0, depth_trunc=1.3, convert_rgb_to_intensity=False
        )

        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)
        pcd.transform(T_world_cam)
        fused_pcd += pcd

    print(f"  [PCD] Успешно объединено кадров: {integrated_frames}")

    if len(fused_pcd.points) > 0:
        # Применяем разреживание точек (шаг 2.5 см)
        fused_pcd = fused_pcd.voxel_down_sample(voxel_size)

        # Агрессивная фильтрация шума (std_ratio=1.0)
        cl, ind = fused_pcd.remove_statistical_outlier(nb_neighbors=40, std_ratio=1.0)
        fused_pcd = fused_pcd.select_by_index(ind)
        fused_pcd.estimate_normals()

    return fused_pcd


def run_reconstruction_pipeline(sync_data, K_small):
    # Разрешение карты глубины
    target_h, target_w = sync_data[0]['small_depth'].shape

    # Оригинальное разрешение цвета
    orig_h, orig_w = sync_data[0]['small_rgb'].shape[:2]

    scale_x = target_w / orig_w
    scale_y = target_h / orig_h

    K_small_scaled = K_small.copy()
    K_small_scaled[0, 0] *= scale_x
    K_small_scaled[1, 1] *= scale_y
    K_small_scaled[0, 2] *= scale_x
    K_small_scaled[1, 2] *= scale_y

    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        target_w, target_h,
        K_small_scaled[0, 0], K_small_scaled[1, 1],
        K_small_scaled[0, 2], K_small_scaled[1, 2]
    )

    # Путь inside_out траектории
    inside_out_path = "data/output/trajectory/inside_out_trajectory.txt"

    if not os.path.exists(inside_out_path):
        print(f"Ошибка: Файл '{inside_out_path}' не найден в рабочей директории.")
        return

    outside_in_traj = load_tum_trajectory(inside_out_path)
    print(f"\n[Загрузка траектории]: Загружено inside_out_path Outside-In: {len(outside_in_traj)}")

    # =========================================================================
    # ГРУППА 1: РЕКОНСТРУКЦИЯ ВООБЩЕ БЕЗ ИСПОЛЬЗОВАНИЯ ТРАЕКТОРИИ (T = I)
    # =========================================================================
    print("\n=== Группа 1: Реконструкция без траектории ===")

    # Вывод 1: TSDF-сетка без траектории
    print("[Вывод 1/4] TSDF Mesh без траектории...")
    mesh_no_traj = reconstruct_tsdf(sync_data, None, intrinsic)
    o3d.io.write_triangle_mesh("data/output/3d/method1_tsdf_mesh_no_trajectory.ply", mesh_no_traj)

    # Вывод 2: Облако точек без траектории
    print("[Вывод 2/4] Point Cloud без траектории...")
    pcd_no_traj = reconstruct_point_cloud_fusion(sync_data, None, intrinsic)
    o3d.io.write_point_cloud("data/output/3d/method2_pcd_fusion_no_trajectory.ply", pcd_no_traj)

    # =========================================================================
    # ГРУППА 2: РЕКОНСТРУКЦИЯ С ИСПОЛЬЗОВАНИЕМ OUTSIDE-IN ТРАЕКТОРИИ
    # =========================================================================
    print("\n=== Группа 2: Реконструкция с Outside-In траекторией ===")

    # Вывод 3: TSDF-сетка с Outside-In траекторией
    print("[Вывод 3/4] TSDF Mesh с Outside-In траекторией...")
    mesh_with_prior = reconstruct_tsdf(sync_data, outside_in_traj, intrinsic)
    o3d.io.write_triangle_mesh("data/output/3d/method1_tsdf_mesh_with_prior.ply", mesh_with_prior)

    # Вывод 4: Облако точек с Outside-In траекторией
    print("[Вывод 4/4] Point Cloud с Outside-In траекторией...")
    pcd_with_prior = reconstruct_point_cloud_fusion(sync_data, outside_in_traj, intrinsic)
    o3d.io.write_point_cloud("data/output/3d/method2_pcd_fusion_with_prior.ply", pcd_with_prior)

    # ВИЗУАЛИЗАЦИЯ

    if len(mesh_no_traj.vertices) > 0:
        print("\nМетод 1: TSDF Mesh без траектории")
        o3d.visualization.draw_geometries([mesh_no_traj], window_name="1/4: TSDF Mesh - NO Trajectory", width=1024,
                                          height=768)

    if len(mesh_with_prior.vertices) > 0:
        print("\nМетод 1: TSDF Mesh с OUTSIDE-IN траекторией")
        o3d.visualization.draw_geometries([mesh_with_prior], window_name="3/4: TSDF Mesh - WITH Outside-In", width=1024,
                                          height=768)

    if len(pcd_no_traj.points) > 0:
        print("\nМетод 2: Point Cloud без траектории")
        o3d.visualization.draw_geometries([pcd_no_traj], window_name="2/4: Point Cloud - NO Trajectory", width=1024,
                                          height=768)


    if len(pcd_with_prior.points) > 0:
        print("\nМетод 2: Point Cloud с OUTSIDE-IN траекторией")
        o3d.visualization.draw_geometries([pcd_with_prior], window_name="4/4: Point Cloud - WITH Outside-In",
                                          width=1024, height=768)
