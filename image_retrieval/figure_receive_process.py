import asyncio
from bleak import BleakClient, BleakScanner
import os
from datetime import datetime
import numpy as np
from retrieval import calculate_color_histogram, compare_histograms
import cv2
import pygame
import threading
import time

# 蓝牙设备的地址和服务UUID
DEVICE_ADDRESS = "CC:8D:A2:0D:82:71"
SERVICE_UUID = "19B10000-E8F2-537E-4F6C-D104768A1214".lower()
CHARACTERISTIC_UUID = "19B10005-E8F2-537E-4F6C-D104768A1214"
# 存储接收到的图像数据
image_data = bytearray()
expected_frame = 0
# 初始化 pygame
pygame.init()
pygame.mixer.init()

# 预先加载的数据和映射
subjects = ['arm','light','dog','printer','tree','picture','extinguisher','sofa','woodenchair','table','chair']
data = np.load("image_retrieval/data.npy")
mapping = {
    'arm': 'sound/robot_arm.mp3',
    'dog': 'sound/dog.mp3',
    'light': 'sound/light.mp3',
    'tree': 'sound/tree.mp3',
    'printer': 'sound/printer.mp3',
    'picture':'sound/picture.mp3',
    'extinguisher':'sound/extinguisher.mp3',
    'sofa':'sound/sofa.mp3',
    'woodenchair':'sound/woodenchair.mp3',
    'table':'sound/table.mp3',
    'chair':'sound/chair.mp3'
}
audio_folder = '../sound'

is_playing = False
play_thread = None

pre_image_hist = None
pre_subject = None

def audio_control(control_queue):
    while True:
        command = input("Enter command (pause, unpause, stop, quit): ").strip().lower()
        if command in ["pause", "unpause", "stop"]:
            control_queue.put(command)
        elif command == "quit":
            control_queue.put("stop")
            break
        else:
            print("Unknown command")

async def play_audio_async_pygame(matches_results, mapping, audio_folder):
    global is_playing, play_thread
    audio_file_path = mapping.get(matches_results)

    if not is_playing:
        is_playing = True

        def play_audio_thread():
            global is_playing
            try:
                print(f"\nPlaying audio: {audio_file_path}")
                pygame.mixer.music.load(audio_file_path)
                pygame.mixer.music.play()

                start_time = time.time()

                while pygame.mixer.music.get_busy():
                    current_time = time.time()
                    if current_time - start_time >= 5:
                        pygame.mixer.music.stop()
                        print("Audio stopped after 5 seconds")
                        break
                    time.sleep(0.1)
            except pygame.error as e:
                print(f"Error playing audio {audio_file_path}: {e}")
            finally:
                is_playing = False

        play_thread = threading.Thread(target=play_audio_thread)
        play_thread.start()

def save_image(image, subject):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S%f')
    file_path = os.path.join('images', f'image_{subject}_{timestamp}.jpg')

    with open(file_path, 'wb') as f:
        f.write(image)

    print(f'Saved image to {file_path}')

def analyze(byte_array, pre_image_hist):
    np_array = np.frombuffer(byte_array, dtype=np.uint8)

    # 将 NumPy 数组解码为图像
    image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    query_histogram = calculate_color_histogram(image)
    if pre_image_hist is not None:
        sim_with_pre = compare_histograms(query_histogram, pre_image_hist)
        print(sim_with_pre)
        if sim_with_pre < 0.2:
            results = []
            for i in range(len(data)):
                similarity = compare_histograms(query_histogram, data[i])
                results.append((i, similarity))
            # 根据相似度排序检索结果
            results.sort(key=lambda x: x[1])
            # 相似度最高的类别
            print(results[0][1])
            if results[0][1] < 0.3:
                id = int(results[0][0] / 30)
                print(subjects[id])
                
                return subjects[id], query_histogram

    return None, query_histogram

async def run_bleak_client():
    async with BleakClient(DEVICE_ADDRESS) as client:
        print("Connected")

        async def callback(sender, data):
            global image_data, expected_frame, pre_subject, pre_image_hist

            frame_number = int.from_bytes(data[:2], byteorder='little')

            if frame_number == 0xFFFF:
                print("End of image data")
                subject, pre_image_hist = analyze(image_data, pre_image_hist)
                if subject is not None:
                    await play_audio_async_pygame(subject, mapping, audio_folder)
                save_image(image_data, subject)
                image_data = bytearray()
                expected_frame = 0
            else:
                if frame_number == expected_frame:
                    image_data.extend(data[2:])
                    expected_frame += 1
                else:
                    print(f"Unexpected frame number {frame_number}, expected {expected_frame}")

        await client.start_notify(CHARACTERISTIC_UUID, callback)

        try:
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            print("Disconnected")

def bleak_thread():
    asyncio.run(run_bleak_client())

def main():
    if not os.path.exists('images'):
        os.makedirs('images')

    # 在单独的线程中运行 BleakClient
    threading.Thread(target=bleak_thread).start()

    # 简单的 pygame 事件循环
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

    pygame.quit()

if __name__ == "__main__":
    main()
