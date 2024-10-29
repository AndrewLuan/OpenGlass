import os
import threading
import time
from datetime import datetime

import cv2
import pyaudio
import pygame
import yaml
from bleak import BleakClient
import numpy as np
import asyncio
from .analyze import analyze
from .bluetooth_earphone import play_audio_async_pygame


def save_image(image, subject):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S%f')
    file_path = './images/' + f'image_{subject}_{timestamp}.jpg'

    with open(file_path, 'wb') as f:
        f.write(image)

    # print(f'Saved image to {file_path}')


# 连接ESP32S3
async def run_bleak_client(config):
    async with BleakClient(config['DEVICE_ADDRESS']) as client:
        print("Connected")

        async def callback(sender, data):
            frame_number = int.from_bytes(data[:2], byteorder='little')

            if frame_number == 0xFFFF:
                # print("End of image data")
                np_array = np.frombuffer(config['image_data'], dtype=np.uint8)
                config['photo'] = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
                subject, config['pre_image_hist'] = analyze(config, False)
                if subject is not None:
                    if not config['is_asking']:
                        await play_audio_async_pygame(config, subject, config['audio_folder'])
                config['pre_subject'] = subject
                save_image(config['image_data'], subject)
                config['image_data'] = bytearray()
                config['expected_frame'] = 0
            else:
                if frame_number == config['expected_frame']:
                    config['image_data'].extend(data[2:])
                    config['expected_frame'] += 1
                else:
                    pass
                    # print(f"Unexpected frame number {frame_number}, expected {config['expected_frame']}")

        await client.start_notify(config['CHARACTERISTIC_UUID'], callback)

        try:
            while not config['is_quit']:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            print("Disconnected")


# 运行连接 ESP32S3
def bleak_client(config):
    asyncio.run(run_bleak_client(config))


# 连接 wifi glass
async def glass_stream(config):
    cap = cv2.VideoCapture(config['rtsp_url'])

    if not cap.isOpened():
        # print("Error: Could not open video stream.")
        exit()

    frame_count = 0
    last_saved_time = time.time()

    while not config['is_quit']:
        ret, frame = cap.read()

        if not ret:
            # print("Error: No more frames to read.")
            break

        # cv2.imshow('RTSP Stream', frame)
        config['photo'] = frame

        # 检查是否达到保存时间间隔
        current_time = time.time()
        if current_time - last_saved_time >= config['interval']:
            # 保存图像
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S%f')
            file_path = './images/' + f'image_{timestamp}.jpg'
            # cv2.imwrite(file_path, frame)
            frame_count += 1
            last_saved_time = current_time  # 更新上次保存时间

            # 分析图片类别
            subject, config['pre_image_hist'] = analyze(config, True)
            if subject is not None:
                if not config['is_asking']:
                    await play_audio_async_pygame(config, subject, config['audio_folder'])
            config['pre_subject'] = subject

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


async def run_glass_retrieval(config):
    last_saved_time = time.time()

    while True:
        current_time = time.time()
        if current_time - last_saved_time >= config['interval'] and config['photo'] is not None:
            # 保存图像
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S%f')
            file_path = os.path.join('images', f'image_{timestamp}.jpg')
            # cv2.imwrite(file_path,global_vars['photo'])

            subject, config['pre_image_hist'] = analyze(config)
            if subject is not None:
                await play_audio_async_pygame(subject, config['audio_folder'])
            last_saved_time = current_time  # 更新上次保存时间


def glass_retrieval(config):
    asyncio.run(run_glass_retrieval(config))


if __name__ == "__main__":
    if not os.path.exists('../images'):
        os.makedirs('../images')

    # Pygame 初始化
    pygame.init()
    pygame.mixer.init()

    # 加载参数文件，存在"config_esp32s3.yaml"中
    # 打开 YAML 文件
    with open('../configs/config_esp32s3.yaml', 'r', encoding='utf-8') as f:
        # 加载 YAML 数据
        config = yaml.safe_load(f)

    # 一些yaml无法存储对象的处理赋值
    config['p'] = pyaudio.PyAudio()
    config['FORMAT'] = eval(config['FORMAT'])
    config['data'] = np.load("../data.npy")
    config['image_data'] = bytearray()
    config['recorded_frames'] = bytearray()

    # 在单独的线程中运行 BleakClient
    threading.Thread(target=bleak_client, args=(config,)).start()

    # 简单的 pygame 事件循环
    running = True
    while not config['is_quit']:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

    pygame.quit()

    # 下面为连接 wifi glass
    # glass_stream(config)
