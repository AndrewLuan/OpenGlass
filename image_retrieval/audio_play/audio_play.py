import json
import os

import cv2
import numpy as np
from playsound import playsound


def play_audio_for_matches(matches_results, mapping, audio_folder):
    """
    播放匹配结果对应的音频文件。

    :param matches_results: 匹配结果列表，每个元素包含 (image_name, score)
    :param mapping: 字典，键是图片名称，值是对应的音频文件名称
    :param audio_folder: 存放音频文件的文件夹路径
    """
    for image_name, _ in matches_results:
        audio_name = mapping.get(image_name)
        if audio_name:
            audio_path = os.path.join(audio_folder, audio_name)
            print(f"Playing audio: {audio_path}")
            playsound(audio_path)
            break  # 只播放匹配结果中的第一个音频文件


# 匹配结果（假设 matches_results 已经有数据）
matches_results = [('image1.jpg', 0.95), ('image2.jpg', 0.85)]

# 映射文件（假设 mapping 已经加载）
# windows需要改play sound的源码 把decode('utf-16')去掉
mapping = {
    'image1.jpg': 'sample-3s.wav',
    'image2.jpg': 'audio2.mp3'
}

# 音频文件夹路径
audio_folder = '../sound/'

# 播放匹配结果的音频
play_audio_for_matches(matches_results, mapping, audio_folder)
