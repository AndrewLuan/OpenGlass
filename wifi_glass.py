import os
import pygame
import yaml
import pyaudio
import numpy as np

from modules.bluetooth_earphone import *
from modules.connect_device import *
from modules.vad_record import *


# Pygame 初始化
pygame.init()
pygame.mixer.init()

# 打开 YAML 文件
with open('configs/config.yaml', 'r', encoding='utf-8') as f:
    # 加载 YAML 数据
    config = yaml.safe_load(f)

config['p'] = pyaudio.PyAudio()
config['FORMAT'] = eval(config['FORMAT'])
config['data'] = np.load("data.npy")
config['image_data'] = bytearray()
config['recorded_frames'] = bytearray()


def handle_user_input():
    """处理用户输入"""
    record_count = 0
    save_thread = None

    while True:
        user_input = input("> ")
        if user_input.lower() == "quit":
            config['record_status'] = False
            config['is_quit'] = True
            if save_thread:
                save_thread.join()
                save_thread = None
            break


if __name__ == "__main__":
    if not os.path.exists('images'):
        os.makedirs('images')

    # 获得蓝牙设备地址
    headset_addr = find_bluetooth_headset(config['target_device'])
    # 如果找得到，就连接
    if headset_addr:
        print("addr", headset_addr)
        record_queue = queue.Queue()  # 创建音频帧队列
        play_queue = queue.Queue()  # 创建音频帧队列

        print("Bluetooth headset found, starting function...")

        # 启动眼镜视频流
        glass_stream_thread = threading.Thread(target=glass_stream, args=(config,))
        glass_stream_thread.start()

        # 启动录音线程
        record_thread = threading.Thread(target=record_audio, args=(config, record_queue))
        record_thread.start()

        # 启动用户输入处理线程
        input_thread = threading.Thread(target=handle_user_input)
        input_thread.start()

        # 等待线程结束
        record_thread.join()
        input_thread.join()
        glass_stream_thread.join()
        config['p'].terminate()

        print("Audio saved to record.wav")
    else:
        print("Failed to find Bluetooth headset.")

