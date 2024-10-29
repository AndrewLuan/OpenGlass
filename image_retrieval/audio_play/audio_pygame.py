import os
import threading
from queue import Queue
import time
import pygame

# 初始化 pygame 的混音器模块
pygame.mixer.init()

def play_audio_for_matches(matches_results, mapping, audio_folder, control_queue):
    """
    播放匹配结果对应的音频文件，并提供控制播放的功能。

    :param matches_results: 匹配结果列表，每个元素包含 (image_name, score)
    :param mapping: 字典，键是图片名称，值是对应的音频文件名称
    :param audio_folder: 存放音频文件的文件夹路径
    :param control_queue: 队列，用于接收控制命令
    """
    for image_name, _ in matches_results:
        audio_name = mapping.get(image_name)
        if audio_name:
            audio_path = os.path.join(audio_folder, audio_name)
            try:
                print(f"\nPlaying audio: {audio_path}")
                pygame.mixer.music.load(audio_path)
                pygame.mixer.music.play()

                while pygame.mixer.music.get_busy() or not control_queue.empty():
                    if not control_queue.empty():
                        command = control_queue.get()
                        if command == "pause":
                            if pygame.mixer.music.get_busy():
                                pygame.mixer.music.pause()
                                print("Audio paused")
                            else:
                                print("No audio is currently playing to pause.")
                        elif command == "unpause":
                            if pygame.mixer.music.get_pos() > 0:
                                pygame.mixer.music.unpause()
                                print("Audio unpaused")
                            else:
                                print("No paused audio to unpause.")
                        elif command == "stop":
                            pygame.mixer.music.stop()
                            print("Audio stopped")
                            break
                    time.sleep(0.1)
            except pygame.error as e:
                print(f"Error playing audio {audio_path}: {e}")
        else:
            print(f"No audio mapping found for image: {image_name}")

def audio_control(control_queue):
    """
    监听用户输入或消息来控制音频播放。

    :param control_queue: 队列，用于发送控制命令
    """
    while True:
        command = input("Enter command (pause, unpause, stop, quit): ").strip().lower()
        if command in ["pause", "unpause", "stop"]:
            control_queue.put(command)
        elif command == "quit":
            control_queue.put("stop")
            break
        else:
            print("Unknown command")

# 匹配结果（假设 matches_results 已经有数据）
matches_results = [('image1.jpg', 0.95), ('image2.jpg', 0.85)]

# 映射文件（假设 mapping 已经加载）
mapping = {
    'image1.jpg': 'sample-3s.wav',
    'image2.jpg': 'audio2.mp3'
}

# 音频文件夹路径
audio_folder = '../sound/'

# 控制队列
control_queue = Queue()

# 启动音频播放线程
audio_thread = threading.Thread(target=play_audio_for_matches, args=(matches_results, mapping, audio_folder, control_queue))
audio_thread.start()

# 启动音频控制
audio_control(control_queue)

# 等待音频播放线程结束
audio_thread.join()
