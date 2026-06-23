# run_pipeline.py (полная версия с Pseudo-GT и 3D визуализацией)
import os
import numpy as np
from align_trajectory import align_and_save_odometry
from evaluate_metrics import print_metrics_table, compute_trajectory_errors
from pseudo_gt import build_and_save_pseudo_gt, calibrate_head_to_sensor, load_tum_trajectory  # Импортируем сборку эталона
from data_loader import BagDataReader
from tracker import estimate_aruco_pose, run_open3d_rgbd_odometry, run_inside_out_localization
from geometry_utils import rvec_tvec_to_T, T_to_TUM_format, calibrate_board_corners_3d
import matplotlib.pyplot as plt

# Путь к папке с вашим rosbag2
BAG_PATH = "data/raw"


def main():
    # Загрузка и подготовка данных, синхронизация кадров
    reader = BagDataReader(BAG_PATH)
    large_raw, small_raw, depth_raw = reader.read_metadata_and_frames()
    full_seq = reader.synchronize_streams(large_raw, small_raw, depth_raw, max_delta_sec=0.03)

    # Разделение на чанки и сохранение
    smooth_move_no_markers_chank = full_seq[351:525]
    rotating_on_one_point_chank = full_seq[563:752]
    fast_rotating_90_deg_with_makers_chank = full_seq[925:1107]
    smooth_moves_with_markers_chank = full_seq[1107:1346]

    # reader.save_video(full_seq, 'data/output/videos/sync_video_full.mp4', target_height=320)
    # reader.save_video(smooth_move_no_markers_chank, 'data/output/videos/sync_video_smooth_moves_no_markers.mp4')
    # reader.save_video(rotating_on_one_point_chank, 'data/output/videos/sync_video_rotating_on_one_point.mp4')
    # reader.save_video(fast_rotating_90_deg_with_makers_chank, 'data/output/videos/sync_video_fast_rotating_90_deg.mp4')
    # reader.save_video(smooth_moves_with_markers_chank, 'data/output/videos/sync_video_smooth_moves_with_markers.mp4')

    # # ==========================================
    # # 3. Калибровка сцены и доски (T_L_B)
    # # ==========================================
    # print("\n[Калибровка] Ищем маркеры доски (20, 21, 23) большой камерой...")
    # stable_board_poses = None
    # T_L_B = None
    # all_T_L_B = []
    #
    # for frame in full_seq:
    #     img = frame['large_rgb']
    #     poses = estimate_aruco_pose(img, reader.K_large, reader.D_large, marker_size=0.09)
    #
    #     if 20 in poses and 21 in poses and 23 in poses:
    #         if stable_board_poses is None:
    #             stable_board_poses = poses
    #
    #         P_20 = poses[20][1].flatten()
    #         P_21 = poses[21][1].flatten()
    #         P_23 = poses[23][1].flatten()
    #
    #         X_B = (P_21 - P_20) / np.linalg.norm(P_21 - P_20)
    #         v1 = P_21 - P_20
    #         v2 = P_23 - P_20
    #         Z_temp = np.cross(v1, v2)
    #         if np.dot(Z_temp, -P_20) < 0:
    #             Z_temp = -Z_temp
    #         Z_B = Z_temp / np.linalg.norm(Z_temp)
    #         Y_B = np.cross(Z_B, X_B)
    #
    #         mat_T = np.eye(4)
    #         mat_T[:3, 0] = X_B
    #         mat_T[:3, 1] = Y_B
    #         mat_T[:3, 2] = Z_B
    #         mat_T[:3, 3] = P_20
    #         all_T_L_B.append(mat_T)
    #
    #         if len(all_T_L_B) >= 30:
    #             break
    #
    # if not all_T_L_B:
    #     print("Ошибка: Калибровочная доска не обнаружена. Пропускаем маркерные траектории.")
    #     return
    #
    # # Усредняем и ортогонализируем T_L_B
    # mean_T_L_B = np.mean(all_T_L_B, axis=0)
    # U, _, Vt = np.linalg.svd(mean_T_L_B[:3, :3])
    # mean_T_L_B[:3, :3] = U @ Vt
    # T_L_B = mean_T_L_B
    # T_B_L = np.linalg.inv(T_L_B)
    # print("-> Положение доски T_B_L успешно рассчитано.")
    #
    # # ==========================================
    # # 4. РАСЧЕТ ТРАЕКТОРИИ 1: Outside-In (Маркер id1)
    # # ==========================================
    # print("\n[Траектория 1] Запуск Outside-In трекинга...")
    # outside_in_lines = []
    # for frame in full_seq:
    #     timestamp = frame['time']
    #     img = frame['large_rgb']
    #     poses = estimate_aruco_pose(img, reader.K_large, reader.D_large, marker_size=0.09)
    #     if 1 in poses:
    #         rvec, tvec = poses[1]
    #         T_L_id1 = rvec_tvec_to_T(rvec, tvec)
    #         T_B_id1 = T_B_L @ T_L_id1
    #         outside_in_lines.append(T_to_TUM_format(T_B_id1, timestamp))
    #
    # out_file_1 = "data/output/trajectory/outside_in_trajectory.txt"
    # with open(out_file_1, "w") as f:
    #     for line in outside_in_lines:
    #         f.write(line + "\n")
    # print(f"-> Сохранено {len(outside_in_lines)} точек в {out_file_1}")
    #
    # # ==========================================
    # # 5. РАСЧЕТ ТРАЕКТОРИИ 2: Inside-Out (Доска стола)
    # # ==========================================
    # board_corners_3d = calibrate_board_corners_3d(T_L_B, stable_board_poses, marker_size=0.09)
    # print(f"-> Откалиброваны 3D углы для {len(board_corners_3d)} маркеров стола.")
    # print("\n[Траектория 2] Запуск Inside-Out самолокализации...")
    # inside_out_lines = run_inside_out_localization(
    #     fast_rotating_90_deg_with_makers_chank+smooth_moves_with_markers_chank, board_corners_3d, reader.K_small, reader.D_small
    # )
    # out_file_2 = "data/output/trajectory/inside_out_trajectory.txt"
    # with open(out_file_2, "w") as f:
    #     for line in inside_out_lines:
    #         f.write(line + "\n")
    # print(f"-> Сохранено {len(inside_out_lines)} точек в {out_file_2}")
    #
    # # ==========================================
    # # СБОРКА PSEUDO GROUND TRUTH (НОВЫЙ ШАГ)
    # # ==========================================
    # print("\n[Pseudo-GT] Сборка референсной траектории-эталона со сглаживанием...")
    # build_and_save_pseudo_gt(
    #     inside_out_path="data/output/trajectory/inside_out_trajectory.txt",
    #     outside_in_path="data/output/trajectory/outside_in_trajectory.txt",
    #     output_path="data/output/trajectory/pseudo_gt_trajectory.txt"
    # )
    #
    #
    # # =========================================================================
    # # 6. ПОСЛЕДОВАТЕЛЬНЫЙ РАСЧЕТ ОДОМЕТРИИ ДЛЯ ВСЕХ 4-Х ЧАНКОВ
    # # =========================================================================
    # # Описываем ваши 4 выделенных чанка
    # chunks_dict = {
    #     "smooth_no_markers": {
    #         "data": smooth_move_no_markers_chank,
    #         "title": "Движение без маркеров"
    #     },
    #     "rotating_on_one_point": {
    #         "data": rotating_on_one_point_chank,
    #         "title": "Вращение на месте"
    #     },
    #     "fast_rotating_90": {
    #         "data": fast_rotating_90_deg_with_makers_chank,
    #         "title": "Быстрый разворот 90° с маркерами"
    #     },
    #     "smooth_with_markers": {
    #         "data": smooth_moves_with_markers_chank,
    #         "title": "Движение с маркерами"
    #     }
    # }
    #
    # print("\n" + "=" * 80)
    # print("ЗАПУСК ВИЗУАЛЬНОЙ ОДОМЕТРИИ (Open3D) ДЛЯ 4-Х ВЫДЕЛЕННЫХ ЧАНКОВ ПО ОЧЕРЕДИ")
    # print("=" * 80)
    #
    # for key, chunk_info in chunks_dict.items():
    #     print(f"\n[Чанк: {chunk_info['title']}] Расчет RGB-D одометрии...")
    #
    #     # А. Считаем сырую одометрию по чанку
    #     odo_lines = run_open3d_rgbd_odometry(chunk_info['data'], reader.K_small)
    #
    #     raw_path = f"data/output/trajectory/camera_only_trajectory_{key}.txt"
    #     with open(raw_path, "w") as f:
    #         for line in odo_lines:
    #             f.write(line + "\n")
    #     print(f"  -> Сырая одометрия сохранена в {raw_path}")
    #
    #     # Б. Выравниваем её методом Umeyama (Sim3 без оценки масштаба) по Pseudo-GT
    #     aligned_path = f"data/output/trajectory/camera_only_aligned_{key}.txt"
    #     print(f"  -> Выравнивание чанка по эталону Pseudo-GT...")
    #     align_and_save_odometry(
    #         odo_path=raw_path,
    #         ref_path="data/output/trajectory/pseudo_gt_trajectory.txt",  # выравниваем по нашему единому Pseudo-GT
    #         output_path=aligned_path
    #     )
    #
    # print("\n" + "=" * 80)
    # print("Расчет и выравнивание одометрии по всем 4-м чанкам успешно завершены!")
    # print("=" * 80)

    # ==========================================
    # 7. ВИЗУАЛИЗАЦИЯ И СРАВНЕНИЕ ТРАЕКТОРИЙ
    # ==========================================
    print("\n[Визуализация] Отрисовка интерактивных графиков сравнения траекторий...")
    file_paths = {
        "Outside-In (маркер id1)": "data/output/trajectory/outside_in_trajectory.txt",
        "Inside-Out (доска стола)": "data/output/trajectory/inside_out_trajectory.txt",
        "Pseudo Ground-Truth (Эталон)": "data/output/trajectory/pseudo_gt_trajectory.txt",

        # 4 трека одометрии по вашим чанкам:
        "VO: Движение без маркеров": "data/output/trajectory/camera_only_aligned_smooth_no_markers.txt",
        "VO: Вращение на месте": "data/output/trajectory/camera_only_aligned_rotating_on_one_point.txt",
        "VO: Быстрый разворот 90°": "data/output/trajectory/camera_only_aligned_fast_rotating_90.txt",
        "VO: Движение с маркерами": "data/output/trajectory/camera_only_aligned_smooth_with_markers.txt"
    }

    colors = ["blue", "red", "black", "#2ca02c", "#8c564b", "#bcbd22", "#17becf"]
    linewidths = [1.8, 1.8, 3.0, 1.8, 1.8, 1.8, 1.8]

    fig = plt.figure(figsize=(13, 10))
    ax = fig.add_subplot(projection='3d')

    # Списки для хранения объектов отрисовки
    plot_lines = []
    plot_scatters = []
    leg_labels = []

    # Загружаем и строим траектории
    for (name, path), color, lw in zip(file_paths.items(), colors, linewidths):
        try:
            data = np.loadtxt(path)
            if data.ndim == 1:
                data = data.reshape(1, -1)

            x = data[:, 1]
            y = data[:, 2]
            z = data[:, 3]

            # Создаем линию траектории
            line, = ax.plot(x, y, z, label=name, color=color, linewidth=lw)
            # Создаем начальную точку
            scatter = ax.scatter(x[0], y[0], z[0], color=color, s=40, edgecolors='black', zorder=5)

            plot_lines.append(line)
            plot_scatters.append(scatter)
            leg_labels.append(name)

        except FileNotFoundError:
            print(f"Предупреждение: Файл '{path}' не найден для отрисовки. Пропускаем.")
        except Exception as e:
            print(f"Не удалось прочитать файл '{path}': {e}")

    # Создаем легенду
    leg = ax.legend(fontsize=9, loc="upper left")

    # Словарь сопоставления элементов легенды и реальных объектов на графике
    lined = {}

    # Делаем элементы легенды кликабельными
    # Пользователь может кликнуть как на цветную линию в легенде, так и на сам текст подписи
    for leg_line, leg_text, orig_line, orig_scatter in zip(leg.get_lines(), leg.get_texts(), plot_lines, plot_scatters):
        leg_line.set_picker(True)
        leg_line.set_pickradius(10)  # радиус клика в пикселях
        leg_text.set_picker(True)

        # Сохраняем связи для обработки клика
        lined[leg_line] = (orig_line, orig_scatter, leg_line, leg_text)
        lined[leg_text] = (orig_line, orig_scatter, leg_line, leg_text)

    def on_pick(event):
        """Обработчик события клика на элемент легенды"""
        if event.artist in lined:
            orig_line, orig_scatter, leg_line, leg_text = lined[event.artist]

            # Меняем видимость объектов на графике
            visible = not orig_line.get_visible()
            orig_line.set_visible(visible)
            orig_scatter.set_visible(visible)

            # Делаем элемент легенды полупрозрачным, если линия выключена
            alpha = 1.0 if visible else 0.2
            leg_line.set_alpha(alpha)
            leg_text.set_alpha(alpha)

            # Перерисовываем холст
            fig.canvas.draw()

    # Подключаем обработчик кликов к интерактивному окну Matplotlib
    fig.canvas.mpl_connect('pick_event', on_pick)

    ax.set_xlabel('X (метры)')
    ax.set_ylabel('Y (метры)')
    ax.set_zlabel('Z (метры)')
    ax.set_title('Интерактивное сравнение траекторий (Кликните по легенде, чтобы скрыть/показать)')
    ax.grid(True)

    print("Интерактивное окно открыто. Кликайте на подписи в легенде для включения/выключения линий.")
    plt.show()

    # =========================================================================
    # 7. ЗАГРУЗКА ТРАЕКТОРИЙ И РАСЧЕТ ОШИБОК ДЛЯ КАЖДОГО ЧАНКА
    # =========================================================================
    print("\n[Анализ] Загрузка траекторий для детального подсчета метрик ATE...")

    gt_path = "data/output/trajectory/pseudo_gt_trajectory.txt"
    inside_out_path = "data/output/trajectory/inside_out_trajectory.txt"
    outside_in_path = "data/output/trajectory/outside_in_trajectory.txt"

    # Загружаем эталоны и маркерные позы
    gt_poses = load_tum_trajectory(gt_path)
    inside_out_poses = load_tum_trajectory(inside_out_path)
    outside_in_poses = load_tum_trajectory(outside_in_path)

    # Указываем пути к 4-м выровненным файлам одометрии по чанкам
    odo_fast_rotating_90_deg_path = "data/output/trajectory/camera_only_aligned_fast_rotating_90.txt"
    odo_rotating_on_one_point_path = "data/output/trajectory/camera_only_aligned_rotating_on_one_point.txt"
    odo_rotating_smooth_no_markers_path = "data/output/trajectory/camera_only_aligned_smooth_no_markers.txt"
    odo_rotating_smooth_with_markers_path = "data/output/trajectory/camera_only_aligned_smooth_with_markers.txt"

    # Загружаем одометрию по чанкам
    odo_fast_rotating_poses = load_tum_trajectory(odo_fast_rotating_90_deg_path)
    odo_rotating_one_point_poses = load_tum_trajectory(odo_rotating_on_one_point_path)
    odo_smooth_no_markers_poses = load_tum_trajectory(odo_rotating_smooth_no_markers_path)
    odo_smooth_with_markers_poses = load_tum_trajectory(odo_rotating_smooth_with_markers_path)

    # Считаем калибровку T_id1_S, чтобы спроецировать Outside-In траекторию на сенсор камеры
    T_id1_S = calibrate_head_to_sensor(inside_out_poses, outside_in_poses)

    # Проецируем Outside-In позы на сенсор камеры S
    outside_in_projected_poses = {}
    for t, T_B_id1 in outside_in_poses.items():
        outside_in_projected_poses[t] = T_B_id1 @ T_id1_S

    # Рассчитываем ошибки ATE для базовых методов
    t_out, err_t_out, err_r_out = compute_trajectory_errors(outside_in_projected_poses, gt_poses)
    t_in, err_t_in, err_r_in = compute_trajectory_errors(inside_out_poses, gt_poses)

    # Рассчитываем ошибки ATE индивидуально для каждого чанка одометрии
    t_odo_no_markers, err_t_odo_no_markers, err_r_odo_no_markers = compute_trajectory_errors(
        odo_smooth_no_markers_poses, gt_poses)
    t_odo_on_point, err_t_odo_on_point, err_r_odo_on_point = compute_trajectory_errors(odo_rotating_one_point_poses,
                                                                                       gt_poses)
    t_odo_fast_rot, err_t_odo_fast_rot, err_r_odo_fast_rot = compute_trajectory_errors(odo_fast_rotating_poses,
                                                                                       gt_poses)
    t_odo_with_markers, err_t_odo_with_markers, err_r_odo_with_markers = compute_trajectory_errors(
        odo_smooth_with_markers_poses, gt_poses)

    # Вывод красивой расширенной Markdown таблицы в консоль
    print("\n" + "=" * 120)
    print(f"| {'Метод оценки / Участок движения':<34} | {'Трансляция (мм)':<41} | {'Вращение (градусы)':<41} |")
    print(
        f"| {'':<34} | {'Mean':^8} | {'Median':^8} | {'p95':^8} | {'RMSE':^8} | {'Mean':^8} | {'Median':^8} | {'p95':^8} | {'RMSE':^8} |")
    print("-" * 120)
    print_metrics_table("Inside-Out (Доска стола)", err_t_in, err_r_in)
    print_metrics_table("Outside-In Projected (id1)", err_t_out, err_r_out)
    print("-" * 120)
    print_metrics_table("VO: Чанк 1 (Вращение на месте)", err_t_odo_on_point, err_r_odo_on_point)
    print_metrics_table("VO: Чанк 2 (Линейное без марк.)", err_t_odo_no_markers, err_r_odo_no_markers)
    print_metrics_table("VO: Чанк 3 (Быстрый разворот 90°)", err_t_odo_fast_rot, err_r_odo_fast_rot)
    print_metrics_table("VO: Чанк 4 (Линейное с маркерами)", err_t_odo_with_markers, err_r_odo_with_markers)
    print("=" * 120 + "\n")

    # =========================================================================
    # ПОСТРОЕНИЕ ГРАФИКОВ ОШИБОК С ОТМЕТКАМИ ЧАНКОВ ВО ВРЕМЕНИ
    # =========================================================================
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 11), sharex=True)

    # 1. Отрисовка базовых методов на обоих графиках
    ax1.plot(t_out, err_t_out, label="Outside-In Projected (id1)", color="blue", alpha=0.5, linewidth=1.2)
    ax1.plot(t_in, err_t_in, label="Inside-Out (Доска стола)", color="red", alpha=0.4, linestyle="--", linewidth=1.2)

    ax2.plot(t_out, err_r_out, label="Outside-In Projected (id1)", color="blue", alpha=0.5, linewidth=1.2)
    ax2.plot(t_in, err_r_in, label="Inside-Out (Доска стола)", color="red", alpha=0.4, linestyle="--", linewidth=1.2)

    # 2. Отрисовка индивидуальных чанков одометрии (разными цветами для наглядности)
    # Чанк 2: Без маркеров
    ax1.plot(t_odo_no_markers, err_t_odo_no_markers, label="VO: Без маркеров", color="#2ca02c", linewidth=2.0)
    ax2.plot(t_odo_no_markers, err_r_odo_no_markers, label="VO: Без маркеров", color="#2ca02c", linewidth=2.0)

    # Чанк 1: Вращение на месте
    ax1.plot(t_odo_on_point, err_t_odo_on_point, label="VO: Вращение на месте", color="#8c564b", linewidth=2.0)
    ax2.plot(t_odo_on_point, err_r_odo_on_point, label="VO: Вращение на месте", color="#8c564b", linewidth=2.0)

    # Чанк 3: Быстрый разворот 90°
    ax1.plot(t_odo_fast_rot, err_t_odo_fast_rot, label="VO: Быстрый разворот 90°", color="#bcbd22", linewidth=2.0)
    ax2.plot(t_odo_fast_rot, err_r_odo_fast_rot, label="VO: Быстрый разворот 90°", color="#bcbd22", linewidth=2.0)

    # Чанк 4: С маркерами
    ax1.plot(t_odo_with_markers, err_t_odo_with_markers, label="VO: С маркерами", color="#17becf", linewidth=2.0)
    ax2.plot(t_odo_with_markers, err_r_odo_with_markers, label="VO: С маркерами", color="#17becf", linewidth=2.0)

    ax1.set_ylabel("Ошибка трансляции ATE (мм)", fontsize=11)
    ax1.set_title("Сравнение ошибок траекторий ATE по выделенным чанкам движения", fontsize=13)
    ax1.grid(True, linestyle=":")
    ax1.legend(fontsize=9, loc="upper left")

    ax2.set_xlabel("Время (секунды)", fontsize=11)
    ax2.set_ylabel("Ошибка вращения ATE (градусы)", fontsize=11)
    ax2.grid(True, linestyle=":")
    ax2.legend(fontsize=9, loc="upper left")

    # Сначала обновим границы графиков, чтобы корректно рассчитать высоту подписей текста
    plt.draw()

    # 2. Определяем границы и параметры отображения для каждого чанка
    chunks_info = [
        {
            "name": "Движение без выраженного \n характера, со скачками кадров",
            "start": full_seq[0]['time'],
            "end": smooth_move_no_markers_chank[0]['time'],
            "color": "gray",
            "alpha": 0.12
        },
        {
            "name": "Движение без маркеров",
            "start": smooth_move_no_markers_chank[0]['time'],
            "end": smooth_move_no_markers_chank[-1]['time'],
            "color": "red",
            "alpha": 0.12
        },
        {
            "name": "Движение без выраженного \n характера, со скачками кадров",
            "start": smooth_move_no_markers_chank[-1]['time'],
            "end": rotating_on_one_point_chank[0]['time'],
            "color": "gray",
            "alpha": 0.12
        },
        {
            "name": "Вращение на месте",
            "start": rotating_on_one_point_chank[0]['time'],
            "end": rotating_on_one_point_chank[-1]['time'],
            "color": "orange",
            "alpha": 0.12
        },
        {
            "name": "Движение без выраженного \n характера, со скачками кадров",
            "start": rotating_on_one_point_chank[-1]['time'],
            "end": fast_rotating_90_deg_with_makers_chank[0]['time'],
            "color": "gray",
            "alpha": 0.12
        },
        {
            "name": "Быстрый разворот 90° c маркерами",
            "start": fast_rotating_90_deg_with_makers_chank[0]['time'],
            "end": fast_rotating_90_deg_with_makers_chank[-1]['time'],
            "color": "green",
            "alpha": 0.12
        },
        {
            "name": "Движение с маркерами",
            "start": smooth_moves_with_markers_chank[0]['time'],
            "end": smooth_moves_with_markers_chank[-1]['time'],
            "color": "purple",
            "alpha": 0.12
        },
        {
            "name": "Движение без выраженного \n характера, со скачками кадров",
            "start": smooth_moves_with_markers_chank[-1]['time'],
            "end": full_seq[-1]['time'],
            "color": "gray",
            "alpha": 0.12
        }
    ]

    # 3. Накладываем цветные фоновые полосы и текстовые аннотации
    for chunk in chunks_info:
        for ax in [ax1, ax2]:
            # Закрашиваем вертикальные полосы на обоих графиках
            ax.axvspan(chunk['start'], chunk['end'], color=chunk['color'], alpha=chunk['alpha'])

        # Добавляем вертикальный текст-подпись по центру каждой полосы на верхнем графике
        # Вычисляем позицию по оси Y (чуть ниже верхней границы графика трансляции)
        y_lim = ax1.get_ylim()
        y_pos = y_lim[1] - (y_lim[1] - y_lim[0]) * 0.08  # 8% ниже верхнего края
        center_x = (chunk['start'] + chunk['end']) / 2

        ax1.text(
            center_x, y_pos, chunk['name'],
            rotation=90,
            ha='center',
            va='top',
            fontsize=8.5,
            fontweight='bold',
            color='black',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.85, edgecolor='none')
        )

    plt.tight_layout()
    plt.show()

    #  ДИАГНОСТИКА СБОЕВ ОДОМЕТРИИ
    print("\n[Диагностика] Запуск оценки здоровья трекинга...")
    from visualize_tracking_health import analyze_and_visualize_tracking_health
    # Запускаем автоматический поиск по всему синхронизированному потоку!
    analyze_and_visualize_tracking_health(full_seq)

    # 3D РЕКОНСТРУКЦИЯ СЦЕНЫ
    print("\n[3D-Реконструкция] Запуск построения 3D моделей...")
    from reconstruct_scene import run_reconstruction_pipeline
    # Запускаем реконструкцию по непрерывному чистому чанку с маркерами
    run_reconstruction_pipeline(smooth_moves_with_markers_chank, reader.K_small)


if __name__ == "__main__":
    main()