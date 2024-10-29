import logging
import base64
import json
import os
import sys
import time
import urllib
import wave
from typing import Any, Dict, List
from urllib.error import URLError
from urllib.parse import quote_plus, urlencode
from urllib.request import Request, urlopen

import gradio as gr
# from transformers import pipeline
import numpy as np
import requests
from aip import AipSpeech
from scipy.io import wavfile
from scipy.signal import resample

APP_ID = " 99081331"
API_KEY = "HI9V7RVRZDPnVq2kkkQFnpTq"
SECRET_KEY = "B2Z0qhoMCcSPd8Y59f2JV79JGfFVdWxN"


def convert_sample_rate(input_path, output_path, target_rate):
    """
    将音频文件的采样率转换为指定的采样率
    :param input_path: 输入文件路径
    :param output_path: 输出文件路径
    :param target_rate: 目标采样率
    """
    # 读取输入文件
    rate, data = wavfile.read(input_path)

    # 如果是立体声，转换为单声道
    if len(data.shape) > 1:
        data = data.mean(axis=1)

    # 计算新的数据长度
    num_samples = round(len(data) * float(target_rate) / rate)

    # 重新采样数据
    new_data = resample(data, num_samples)

    # 确保数据是整数
    new_data = new_data.astype(np.int16)

    # 保存到输出文件
    with wave.open(output_path, 'w') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16位音频
        wf.setframerate(target_rate)
        wf.writeframes(new_data.tobytes())
    return new_data.tobytes()


def get_file_content_as_base64(path, urlencoded=False):
    """
    获取文件base64编码
    :param path: 文件路径
    :param urlencoded: 是否对结果进行urlencoded
    :return: base64编码信息
    """
    with open(path, "rb") as f:
        content = base64.b64encode(f.read()).decode("utf8")
        if urlencoded:
            content = urllib.parse.quote_plus(content)
    return content


def get_file_length(path):
    """
    获取文件长度（字节数）
    :param path: 文件路径
    :return: 文件长度
    """
    with open(path, "rb") as f:
        return len(f.read())
    # f.close()


def get_access_token():
    """
    使用 AK，SK 生成鉴权签名（Access Token）
    :return: access_token，或是None(如果错误)
    """
    url = "https://aip.baidubce.com/oauth/2.0/token"
    params = {"grant_type": "client_credentials",
              "client_id": API_KEY, "client_secret": SECRET_KEY}
    return str(requests.post(url, params=params).json().get("access_token"))


def audio2text(input_path):
    # 获取文件所在文件夹和文件名
    folder_path, filename = os.path.split(input_path)
    name, ext = os.path.splitext(filename)
    # 定义输出文件夹和输出路径
    output_folder = os.path.join(os.path.dirname(folder_path), "output")
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, f"{name}_output{ext}")

    url = "https://vop.baidu.com/server_api"
    # 如果输出文件已经存在，跳过处理
    if os.path.exists(output_path):
        print(f"文件 {output_path} 已经存在，跳过处理。")
        with open(output_path, "rb") as f:
            trans_audio_data = f.read()
    else:
        # 降采样
        trans_audio_data = convert_sample_rate(input_path, output_path, 16000)

    # trans_audio_data = convert_sample_rate(audio_path, output_path, 16000)
    audio_length = len(trans_audio_data)
    audio_content = base64.b64encode(trans_audio_data).decode("utf8")
    print("转换采样率完成")
    payload = json.dumps({
        "format": "wav",
        "rate": 16000,
        "channel": 1,
        "cuid": "aPKg3eeLhncy2WzuJIl0Bk19HQbTfnEV",
        "token": get_access_token(),
        "speech": audio_content,
        "len": audio_length
    })
    headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json'
    }

    response = requests.request("POST", url=url, headers=headers, data=payload)
    result = response.json()
    print(result)
    if 'err_no' in result and result['err_no'] == 0:
        return result['result'][0]
    else:
        raise Exception(f"语音识别错误: {result}")


def text2audioSDK(text):
    audio_client = AipSpeech(APP_ID, API_KEY, SECRET_KEY)

    # Synthesis parameters
    options = {
        'spd': 5,  # Speed
        'vol': 5,  # Volume
        'pit': 5,  # Pitch
        'per': 1,  # Speaker ID
        'aue': 6   # Audio format
    }

    try:
        result = audio_client.synthesis(text, 'zh', 1, options)
        # Check if the result is binary audio data
        if not isinstance(result, dict):
            with open('audio.mp3', 'wb') as f:
                f.write(result)
            print("Audio file has been successfully saved as audio.mp3")
        else:
            # The result is a dictionary, indicating an error
            print(f"Text-to-speech synthesis failed, error info: {result}")
    except Exception as e:
        print(f"An exception occurred during text-to-speech synthesis: {e}")


def text2audio(text):
    # 短文本，不超过60个汉字
    # TTS参数
    PER = 1  # 发音人编号
    SPD = 5  # 语速
    PIT = 5  # 音调
    VOL = 5  # 音量
    AUE = 6  # 采样率
    FORMAT = "wav"  # 输出格式
    cuid = "aPKg3eeLhncy2WzuJIl0Bk19HQbTfnEVdddada"  # 用户唯一标识
    url = "https://tsn.baidu.com/text2audio"

    # 获取访问令牌
    token = get_access_token()
    if not token:
        print("Failed to get access token")
        return

    # 对文本进行URL编码
    tex = quote_plus(text)

    # 准备请求参数
    payload = f'tex={tex}&tok={token}&cuid={cuid}&ctp=1&lan=zh&spd={SPD}&pit={PIT}&vol={VOL}&per={PER}&aue={AUE}'

    headers = {
        'Content-Type': 'application/x-www-form-urlencoded',
        'Accept': '*/*'
    }

    try:
        # 发送POST请求
        response = requests.post(url, headers=headers, data=payload)
        # 如果合成成功，返回的Content-Type以“audio”开头
        # aue =3 ，返回为二进制mp3文件，具体header信息 Content-Type: audio/mp3；
        # aue =4 ，返回为二进制pcm文件，具体header信息 Content-Type:audio/basic;codec=pcm;rate=16000;channel=1
        # aue =5 ，返回为二进制pcm文件，具体header信息 Content-Type:audio/basic;codec=pcm;rate=8000;channel=1
        # aue =6 ，返回为二进制wav文件，具体header信息 Content-Type: audio/wav；rate=8000；channel=1

        # 检查响应是否包含音频数据
        if response.headers['Content-Type'].startswith('audio'):
            with open('result.' + FORMAT, 'wb') as f:
                f.write(response.content)
            print("Result saved as: result." + FORMAT)
        else:
            print("Error in TTS API:", response.text)

    except requests.RequestException as e:
        print(f"An error occurred while making the request: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


def test_all_audio_files_in_folder(input_folder):
    results = {}
    for filename in os.listdir(input_folder):
        if filename.endswith(".wav") and not filename.endswith("_output.wav"):
            file_path = os.path.join(input_folder, filename)
            try:
                result = audio2text(file_path)
                results[filename] = result
            except Exception as e:
                results[filename] = str(e)
    return results


def process_audio_files_in_folder(input_folder):
    """
    处理指定文件夹中的所有音频文件，并打印结果
    :param input_folder: 输入文件夹路径
    """
    # 创建输入文件夹（如果不存在）
    os.makedirs(input_folder, exist_ok=True)

    results = test_all_audio_files_in_folder(input_folder)
    for filename, result in results.items():
        print(f"{filename}: {result}")


def test_text2audio():
    text = "测试"
    text2audio(text)
    text2audioSDK(text)


if __name__ == '__main__':
    # audio_path = "./audio/test.wav"
    # print(audio2text(audio_path))
    # input_folder = "./audio/input"  # 输入文件夹
    # process_audio_files_in_folder(input_folder)

    input_audio = gr.Audio(sources=["microphone", "upload"], type="filepath")
    input_text = gr.Textbox(lines=2, label="请输入文本")
    # input_audio.change(audio2text, input_audio, input_text)
    demo = gr.Interface(
        audio2text,
        [input_audio],
        "text",
    )
    
    demo.launch()
