import numpy as np
import cv2
from pathlib import Path
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore


class BagDataReader:
    def __init__(self, bag_path_str: str):
        self.bag_path = Path(bag_path_str)
        self.typestore = get_typestore(Stores.LATEST)

        # Топики из вашего metadata.yaml
        self.TOPIC_SMALL_COLOR = "/camera_small/color/image_raw/compressed"
        self.TOPIC_BIG_COLOR = "/camera_big/color/image_raw/compressed"
        self.TOPIC_SMALL_DEPTH = "/camera_small/depth/image_raw"
        self.TOPIC_SMALL_INFO = "/camera_small/color/camera_info"
        self.TOPIC_BIG_INFO = "/camera_big/color/camera_info"

        # Переменные для калибровок
        self.K_large, self.D_large = None, None
        self.K_small, self.D_small = None, None

    def read_metadata_and_frames(self):
        """
        Проходит по bag-файлу, извлекает параметры калибровок (K, D)
        и собирает списки кадров.
        """
        large_frames = []
        small_frames = []
        depth_frames = []

        print("Начинаем чтение bag-файла...")

        with AnyReader([self.bag_path], default_typestore=self.typestore) as reader:
            for connection, timestamp, rawdata in reader.messages():
                topic = connection.topic
                msgtype = connection.msgtype

                try:
                    # Десериализуем сообщение
                    msg = reader.deserialize(rawdata, msgtype)

                    # Извлекаем header-таймстемп
                    # В ROS2 stamp содержит sec и nanosec
                    time_sec = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

                    # 1. Считываем калибровки для большой камеры
                    if topic == self.TOPIC_BIG_INFO and self.K_large is None:
                        self.K_large = np.array(msg.k, dtype=np.float32).reshape(3, 3)
                        self.D_large = np.array(msg.d, dtype=np.float32)
                        print("Успешно считана калибровка большой камеры!")

                    # 2. Считываем калибровки для малой камеры
                    elif topic == self.TOPIC_SMALL_INFO and self.K_small is None:
                        self.K_small = np.array(msg.k, dtype=np.float32).reshape(3, 3)
                        self.D_small = np.array(msg.d, dtype=np.float32)
                        print("Успешно считана калибровка малой камеры!")

                    # 3. Декодируем сжатые цветные кадры большой камеры
                    elif topic == self.TOPIC_BIG_COLOR:
                        np_arr = np.frombuffer(msg.data, dtype=np.uint8)
                        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                        large_frames.append({"time": time_sec, "image": img})

                    # 4. Декодируем сжатые цветные кадры малой камеры
                    elif topic == self.TOPIC_SMALL_COLOR:
                        np_arr = np.frombuffer(msg.data, dtype=np.uint8)
                        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                        small_frames.append({"time": time_sec, "image": img})

                    # 5. Декодируем несжатые кадры глубин малой камеры
                    elif topic == self.TOPIC_SMALL_DEPTH:
                        # ТЗ указывает на RGB-D малую камеру. Глубина обычно 16-битная (mono16 / 16UC1)
                        depth_raw = np.frombuffer(msg.data, dtype=np.uint16)
                        depth_img = depth_raw.reshape(msg.height, msg.width)
                        depth_frames.append({"time": time_sec, "image": depth_img})

                except Exception as e:
                    # Оборонительное чтение: логгируем и идем дальше
                    print(f"Ошибка декодирования топика {topic} на времени {timestamp}: {e}")
                    continue

        return large_frames, small_frames, depth_frames

    def synchronize_streams(self, large_frames, small_frames, depth_frames, max_delta_sec=0.03):
        """
        Синхронизирует потоки кадров большой камеры, малой камеры и карты глубин.
        """
        print("Синхронизация потоков по времени...")
        synchronized = []

        # Для быстрого поиска преобразуем времена в массивы numpy
        large_times = np.array([f['time'] for f in large_frames])
        depth_times = np.array([d['time'] for d in depth_frames])

        for sf in small_frames:
            t_small = sf['time']

            # Ищем ближайший по времени кадр глубины малой камеры
            if len(depth_times) == 0:
                continue
            idx_depth = np.argmin(np.abs(depth_times - t_small))
            if np.abs(depth_times[idx_depth] - t_small) > max_delta_sec:
                continue

            # Ищем ближайший по времени кадр большой камеры
            if len(large_times) == 0:
                continue
            idx_large = np.argmin(np.abs(large_times - t_small))
            if np.abs(large_times[idx_large] - t_small) > max_delta_sec:
                continue

            synchronized.append({
                'time': t_small,
                'small_rgb': sf['image'],
                'small_depth': depth_frames[idx_depth]['image'],
                'large_rgb': large_frames[idx_large]['image']
            })

        print(f"Синхронизация завершена. Успешно сопоставлено кадров: {len(synchronized)}")
        return synchronized

    def save_video(
        self, sync_data, output_path="output.mp4", fps=30.0, target_height=480
    ):
        """Сохранение синхронизированных кадров в один видеофайл в виде горизонтальной склейки.

        sync_data: список словарей с кадрами и метаданными
        output_path: путь для сохранения итогового видео (.mp4)
        fps: частота кадров видеофайла
        target_height: базовая высота для ресайза кадров
        """
        if not sync_data:
            print("Список синхронизированных данных пуст. Запись отменена.")
            return

        print(f"Начало записи видео в файл: {output_path}...")

        num_frames = len(sync_data)
        video_writer = None

        for frame_idx, frame in enumerate(sync_data):
            # 1. Получаем исходные изображения
            large_img = frame["large_rgb"].copy()
            small_img = frame["small_rgb"].copy()
            depth_img = frame["small_depth"].copy()
            timestamp = frame["time"]

            # 2. Обрабатываем карту глубин (из 16-бит в цветное 8-бит)
            valid_depth_mask = depth_img > 0
            depth_colored = np.zeros_like(small_img, dtype=np.uint8)

            if np.any(valid_depth_mask):
                depth_clipped = np.clip(depth_img, 0, 3000)
                depth_normalized = cv2.normalize(
                    depth_clipped, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
                )
                depth_colored = cv2.applyColorMap(
                    depth_normalized, cv2.COLORMAP_JET
                )
                depth_colored[~valid_depth_mask] = 0

            # Вспомогательная функция для изменения размера
            def resize_to_height(img, height):
                h, w = img.shape[:2]
                aspect_ratio = w / h
                new_width = int(height * aspect_ratio)
                return cv2.resize(img, (new_width, height))

            # 3. Приводим все изображения к одной высоте
            large_resized = resize_to_height(large_img, target_height)
            small_resized = resize_to_height(small_img, target_height)
            depth_resized = resize_to_height(depth_colored, target_height)

            # Добавляем текстовые подписи
            cv2.putText(
                large_resized,
                "Large Camera (Outside-in)",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                small_resized,
                "Small Camera (Inside-out)",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                depth_resized,
                f"Small Depth | Frame: {frame_idx}/{num_frames}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                depth_resized,
                f"TS: {timestamp:.3f} s",
                (10, target_height - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1,
            )

            # 4. Склеиваем кадры горизонтально
            combined_view = np.hstack([large_resized, small_resized, depth_resized])

            # 5. Ленивая инициализация cv2.VideoWriter
            # Мы делаем это здесь, так как финальное разрешение (ширина) зависит от aspect_ratio всех камер
            if video_writer is None:
                height, width = combined_view.shape[:2]
                fourcc = cv2.VideoWriter.fourcc(*"mp4v")
                video_writer = cv2.VideoWriter(
                    output_path, fourcc, fps, (width, height)
                )

            # 6. Записываем текущую склейку в видеофайл
            video_writer.write(combined_view)

        # Освобождаем память и закрываем файл
        if video_writer is not None:
            video_writer.release()

        print(f"Видео успешно сохранено: {output_path}")
