import time

import pyaudio  # 替换 sounddevice
import wave
import numpy as np
import bluetooth  # 用于蓝牙扫描和连接
import threading
import queue
from t2s2t_pipeline.user2llm import chat
import os
from dotenv import load_dotenv
from openai import OpenAI
from t2s2t_pipeline.text2speech import text2speech
from t2s2t_pipeline.speech2text import speech2text

from deepgram import (
    DeepgramClient,
    PrerecordedOptions,
    FileSource,
)


target_device = "HUAWEI FreeBuds Pro"
# 设置音频参数（根据需要修改）
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1  # 假设立体声
RATE = 44100
RECORD_SECONDS = 5

# pyaudio 初始化
p = pyaudio.PyAudio()
record_status = True


def find_bluetooth_headset(device_name):
    """搜索蓝牙耳机并返回设备地址"""
    nearby_devices = bluetooth.discover_devices(lookup_names=True)
    print(nearby_devices)
    for addr, name in nearby_devices:
        if device_name in name:  # 根据实际耳机名称修改
            return addr
    return None


def play_audio(headset_address, device_name, filename, audio_queue):
    global record_status
    # 查找蓝牙耳机麦克风输出设备索引
    for i in range(p.get_device_count()):
        dev_info = p.get_device_info_by_index(i)
        if device_name in str(dev_info['name']) and dev_info['maxOutputChannels'] > 0:
            output_device_index = i
            break
    else:
        raise Exception("Bluetooth headset output not found")

    """播放指定的音频文件"""
    wf = wave.open(filename, 'rb')

    stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                    channels=wf.getnchannels(),
                    rate=wf.getframerate(),
                    output=True,
                    output_device_index=output_device_index)

    data = wf.readframes(CHUNK)
    while len(data) > 0:
        stream.write(data)
        audio_queue.put(data)  # 将音频数据放入队列
        data = wf.readframes(CHUNK)

    stream.stop_stream()
    stream.close()
    wf.close()


def record_audio(headset_address, device_name, audio_queue):
    """录制蓝牙耳机音频并保存为 WAV 文件"""
    global record_status, is_quit, start_record, stop_record

    # 查找蓝牙耳机麦克风输入设备索引
    for i in range(p.get_device_count()):
        dev_info = p.get_device_info_by_index(i)
        if device_name in str(dev_info['name']) and dev_info['maxInputChannels'] > 0:
            input_device_index = i
            break
    else:
        raise Exception("Bluetooth headset microphone not found")

    def callback(in_data, frame_count, time_info, status):
        audio_queue.put(in_data)
        return None, pyaudio.paContinue

    while not is_quit:
        if record_status:
            # 打开音频流
            stream = p.open(format=FORMAT,
                            channels=CHANNELS,
                            rate=RATE,
                            input=True,
                            frames_per_buffer=CHUNK,
                            input_device_index=input_device_index,
                            stream_callback=callback)  # 指定输入设备索引

            print("* recording")
            stream.start_stream()

            # 等待外部停止信号
            while stream.is_active() and record_status:
                pass

            print("* done recording")

            # 停止和关闭流
            stream.stop_stream()
            stream.close()
        else:
            time.sleep(1)


def save_audio_stream(audio_queue, save_filename):
    """从队列中获取音频帧并保存为 WAV 文件"""
    global record_status, recorded_frames, recorded_wav
    wf = wave.open(save_filename, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    recorded_frames.clear()
    gain = 3

    while True:
        try:
            data = audio_queue.get_nowait()
            # 将音频数据转换为 NumPy 数组
            audio_array = np.frombuffer(data, dtype=np.int16)
            # 应用增益
            audio_array = audio_array * gain
            # 限制幅度在有效范围内（-32768 到 32767）
            audio_array = np.clip(audio_array, -32768, 32767)
            # 将 NumPy 数组转换回字节串
            data = audio_array.astype(np.int16).tobytes()
            recorded_frames.append(data)
            wf.writeframes(data)
        except queue.Empty:
            if not record_status:  # 录音结束
                break

    wf.close()
    recorded_wav = b''.join(recorded_frames)


def get_recorded_audio(audio_queue):
    """从缓存中获取录制的音频数据"""
    global recorded_frames
    while not audio_queue.empty():
        recorded_frames.append(audio_queue.get_nowait())
    return b''.join(recorded_frames)


def chat_with_llm():
    load_dotenv()

    AUDIO_FILE = "./audio/ask.wav"
    OUTPUT_FILE = "./audio/answer.wav"

    DG_API_KEY = '1fda42ddec0455802e88971738e46aa9fc32e3c7'
    KIMI_API_KEY = "sk-uSS6OrqyCA4XSdUXiPDQyaX5M6wZhwNG6etQx6rUja6IMm47"
    client = OpenAI(
        api_key=KIMI_API_KEY,
        base_url="https://api.moonshot.cn/v1",
    )

    history = [
        {"role": "system", "content": f'''
                You are an assistant for question-answering tasks. Use the following pieces of retrieved context and the conversation history to continue the conversation.
                If you don't know the answer, just say that you didn't find any related information or you that don't know. Use three sentences maximum and keep the answer concise.
                If the message doesn't require context, it will be empty, so answer the question casually. Answer in English.
                '''}
    ]
    deepgram = DeepgramClient(DG_API_KEY)
    user_input = speech2text(AUDIO_FILE, deepgram)
    llm_output = chat(client, user_input, history)
    text_option = {"text": llm_output}
    text2speech(OUTPUT_FILE, deepgram, text_option)


def handle_user_input(record_queue, play_queue):
    """处理用户输入"""
    global record_status, is_quit, recorded_frames
    record_count = 0
    save_thread = None

    while True:
        user_input = input("> ")
        if user_input.lower() == "quit":
            record_status = False
            is_quit = True
            if save_thread:
                save_thread.join()
                save_thread = None
            break
        elif user_input.lower() == "start record":
            # if not record_status:  # 只有在未录音状态下才能开始录音
            record_status = True
            record_count = record_count + 1
            save_thread = threading.Thread(target=save_audio_stream, args=(record_queue, f"./audio/ask.wav"))
            save_thread.start()
        elif user_input.lower() == "stop record":
            record_status = False
            save_thread.join()
            save_thread = None
            print(f"Recorded audio length: {len(recorded_wav)} bytes")

            # 停止录音后，提示用户输入要播放的音频文件
            chat_with_llm()
            filename = "./audio/answer.wav"
            # filename = "test.wav"
            play_thread = threading.Thread(target=play_audio, args=(headset_addr, target_device, filename, play_queue))
            play_thread.start()
            play_thread.join()  # 等待播放线程结束
            play_thread = None
        elif user_input.lower() == "stop play":
            play_queue.put(None)  # 向队列中放入 None 表示停止播放


if __name__ == "__main__":
    headset_addr = find_bluetooth_headset(target_device)
    if headset_addr:
        print("addr", headset_addr)
        record_queue = queue.Queue()  # 创建音频帧队列
        play_queue = queue.Queue()  # 创建音频帧队列
        record_status = False  # 录音状态标志
        is_quit = False
        recorded_frames = []
        recorded_wav = None
        print("Bluetooth headset found, starting audio recording...")

        # 启动录音线程
        record_thread = threading.Thread(target=record_audio, args=(headset_addr, target_device, record_queue))
        record_thread.start()

        # 启动用户输入处理线程
        input_thread = threading.Thread(target=handle_user_input, args=(record_queue, play_queue))
        input_thread.start()

        # 等待线程结束
        record_thread.join()
        input_thread.join()
        p.terminate()

        print("Audio saved to record.wav")
    else:
        print("Failed to find Bluetooth headset.")
