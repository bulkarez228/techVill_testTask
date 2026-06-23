# visualize_tracking_health.py
import cv2
import numpy as np


def analyze_and_visualize_tracking_health(sync_data):
    """
    Автоматически находит пары кадров с лучшим и худшим числом совпадений ORB,
    рисует линии связей и сохраняет диагностические изображения.
    """
    if len(sync_data) < 2:
        print("Недостаточно кадров для анализа.")
        return

    print("\n[Диагностика] Сканируем кадры для оценки качества сопоставления...")

    orb = cv2.ORB_create(nfeatures=500)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    max_matches_count = -1
    min_matches_count = 999999

    best_pair_data = None
    worst_pair_data = None

    for i in range(len(sync_data) - 1):
        frame_src = sync_data[i]
        frame_dst = sync_data[i + 1]

        img1 = frame_src['small_rgb']
        img2 = frame_dst['small_rgb']

        kp1, des1 = orb.detectAndCompute(img1, None)
        kp2, des2 = orb.detectAndCompute(img2, None)

        if des1 is None or des2 is None:
            continue

        # Находим сырые совпадения
        matches = bf.match(des1, des2)

        # Отфильтровываем только надежные, качественные совпадения (расстояние < 45.0)
        good_matches = [m for m in matches if m.distance < 45.0]
        good_matches = sorted(good_matches, key=lambda x: x.distance)
        count = len(good_matches)

        # Ищем абсолютный максимум по качеству трекинга
        if count > max_matches_count:
            max_matches_count = count
            best_pair_data = {
                'frame_idx': i,
                'frame_src': frame_src,
                'frame_dst': frame_dst,
                'kp1': kp1,
                'kp2': kp2,
                'matches': good_matches
            }

        # Ищем абсолютный минимум по качеству трекинга (исключая нулевые затыки)
        if count < min_matches_count and count > 3:
            min_matches_count = count
            worst_pair_data = {
                'frame_idx': i,
                'frame_src': frame_src,
                'frame_dst': frame_dst,
                'kp1': kp1,
                'kp2': kp2,
                'matches': good_matches
            }

    # =========================================================================
    # РИСУЕМ РЕЗУЛЬТАТЫ
    # =========================================================================

    # 1. Отрисовка лучшего совпадения
    if best_pair_data is not None:
        best_idx = best_pair_data['frame_idx']
        print(f"-> Лучший трекинг найден на кадре {best_idx} ({max_matches_count} надежных связей)")

        best_img = cv2.drawMatches(
            best_pair_data['frame_src']['small_rgb'], best_pair_data['kp1'],
            best_pair_data['frame_dst']['small_rgb'], best_pair_data['kp2'],
            best_pair_data['matches'][:40], None,  # рисуем топ-40 лучших связей для красоты
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
            matchColor=(0, 255, 0)  # Зеленые линии для идеального трекинга
        )

        title_best = f"HEALTHY TRACKING: Frame {best_idx} to {best_idx + 1} ({max_matches_count} matches)"
        cv2.putText(best_img, title_best, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.imwrite("content_for_readme/diagnostic_best_tracking_health.png", best_img)
        print("   -> Сохранено: diagnostic_best_tracking_health.png")
        cv2.imshow("Best Tracking Quality", best_img)

    # 2. Отрисовка худшего совпадения
    if worst_pair_data is not None:
        worst_idx = worst_pair_data['frame_idx']
        print(f"-> Худший трекинг найден на кадре {worst_idx} ({min_matches_count} надежных связей)")

        worst_img = cv2.drawMatches(
            worst_pair_data['frame_src']['small_rgb'], worst_pair_data['kp1'],
            worst_pair_data['frame_dst']['small_rgb'], worst_pair_data['kp2'],
            worst_pair_data['matches'][:40], None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
            matchColor=(0, 0, 255)  # Красные линии для слабого трекинга
        )

        title_worst = f"WEAK TRACKING: Frame {worst_idx} to {worst_idx + 1} ({min_matches_count} matches)"
        cv2.putText(worst_img, title_worst, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        cv2.imwrite("content_for_readme/diagnostic_worst_tracking_health.png", worst_img)
        print("   -> Сохранено: diagnostic_worst_tracking_health.png")
        cv2.imshow("Worst Tracking Quality", worst_img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()