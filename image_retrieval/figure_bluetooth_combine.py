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
import argparse
import queue
from t2s2t_pipeline.user2llm import chat
from dotenv import load_dotenv
from openai import OpenAI
from t2s2t_pipeline.text2speech import text2speech
from t2s2t_pipeline.speech2text import speech2text
import pyaudio  # 替换 sounddevice
import wave
import bluetooth  # 用于蓝牙扫描和连接
import collections
import webrtcvad
import yaml
from vad import Audio, VADAudio, Int2Float
from halo import Halo
import contextlib
import torch
import torchaudio

from deepgram import (
    DeepgramClient,
    PrerecordedOptions,
    FileSource,
)

# Pygame 初始化
pygame.init()
pygame.mixer.init()

# 打开 YAML 文件
with open('config.yaml', 'r', encoding='utf-8') as f:
    # 加载 YAML 数据
    config = yaml.safe_load(f)

config['p'] = pyaudio.PyAudio()
config['FORMAT'] = eval(config['FORMAT'])
config['data'] = np.load("data.npy")
config['image_data'] = bytearray()
config['recorded_frames'] = bytearray()


def find_input_device_idx(device_name):
    # 查找蓝牙耳机麦克风输入设备索引
    for i in range(config['p'].get_device_count()):
        dev_info = config['p'].get_device_info_by_index(i)
        if device_name in str(dev_info['name']) and dev_info['maxInputChannels'] > 0:
            config['input_device_idx'] = i
            break
    else:
        raise Exception("Bluetooth headset microphone not found")


def find_output_device_idx(device_name):
    # 查找蓝牙耳机麦克风输出设备索引
    for i in range(config['p'].get_device_count()):
        dev_info = config['p'].get_device_info_by_index(i)
        if device_name in str(dev_info['name']) and dev_info['maxOutputChannels'] > 0:
            config['output_device_idx'] = i
            break
    else:
        raise Exception("Bluetooth headset output not found")


def find_bluetooth_headset(device_name):
    """搜索蓝牙耳机并返回设备地址"""
    nearby_devices = bluetooth.discover_devices(lookup_names=True)
    print(nearby_devices)
    for addr, name in nearby_devices:
        if device_name in name:  # 根据实际耳机名称修改
            return addr
    return None


def play_audio(headset_address, device_name, filename, audio_queue):
    """播放指定的音频文件"""
    wf = wave.open(filename, 'rb')

    stream = config['p'].open(format=config['p'].get_format_from_width(wf.getsampwidth()),
                              channels=wf.getnchannels(),
                              rate=wf.getframerate(),
                              output=True,
                              output_device_index=config['output_device_idx'])

    data = wf.readframes(config['CHUNK'])
    while len(data) > 0:
        stream.write(data)
        audio_queue.put(data)  # 将音频数据放入队列
        data = wf.readframes(config['CHUNK'])

    stream.stop_stream()
    stream.close()
    wf.close()


def play_audio_pygame(headset_address, device_name, filename, audio_queue):
    # Pygame 初始化，包括音频模块
    pygame.mixer.init()

    # 1. 加载音频文件
    sound = pygame.mixer.Sound(filename)

    # 2. (可选) 指定输出设备 (Pygame 1.9.6+)
    # 查找设备索引的逻辑与 PyAudio 类似，但 Pygame 的 API 不同
    # for i in range(pygame.mixer.get_num_devices()):
    #     dev_info = pygame.mixer.get_device_info(i)
    #     if device_name in dev_info.name and dev_info.is_output:
    #         output_device_index = i
    #         pygame.mixer.set_device(output_device_index)
    #         break
    # else:
    #     raise Exception("Bluetooth headset output not found")

    # 3. 播放音频
    channel = sound.play()

    # 4. 将音频数据放入队列（需要解码）
    while channel.get_busy():  # 循环直到播放结束
        # 注意：Pygame 不直接提供原始音频数据，需要解码
        # 这里假设您需要 PCM 格式的数据，具体解码方式取决于您的需求
        # 下面是一个简化的示例，可能需要根据实际情况调整
        samples = pygame.sndarray.array(sound)  # 获取音频样本数组
        audio_queue.put(samples.tobytes())  # 将样本转换为字节流放入队列

    # 5. (可选) 释放资源
    sound.stop()  # 停止播放 (如果需要)
    pygame.mixer.quit()  # 退出 Pygame 音频模块


def vad_stream(headset_address, device_name, audio_queue, frame_duration_ms=20, aggressiveness=3, callback=None):
    """
    使用pyaudio Stream进行实时音频流语音活动检测

    参数：
        pa (pyaudio.PyAudio): PyAudio对象
        sample_rate (int): 音频采样率
        frame_duration_ms (int): VAD帧时长，通常为10, 20, 30 ms
        aggressiveness (int): VAD灵敏度级别 (0, 1, 2, 3)，数值越大越严格
        callback (function): 回调函数，用于处理VAD结果
    """
    vad = webrtcvad.Vad(aggressiveness)

    config['CHUNK'] = int(config['RATE'] * frame_duration_ms / 1000) * 2  # 16-bit PCM
    ring_buffer = collections.deque(maxlen=int(200 / frame_duration_ms))

    def stream_callback(in_data, frame_count, time_info, status):
        is_speech = vad.is_speech(in_data, config['RATE'])
        print("is speech", is_speech)

        ring_buffer.append((in_data, is_speech))
        num_voiced = len([f for f, speech in ring_buffer if speech])

        if num_voiced > 0.9 * ring_buffer.maxlen:
            config['vad_triggered'] = True
            if not config['record_status']:
                print("* recording")
                config['record_status'] = True
        elif num_voiced < 0.3 * ring_buffer.maxlen:
            config['vad_triggered'] = False
            if config['record_status']:
                print("* stop recording")
                config['record_status'] = False
                save_audio_stream(audio_queue, f"audio/record.wav")

        if config['record_status']:
            audio_queue.put(in_data)  # 存储录音数据

        return None, pyaudio.paContinue  # 返回None表示不需要输出音频

    # while not config['is_quit']:
    stream = config['p'].open(format=config['FORMAT'],
                              channels=config['CHANNELS'],
                              rate=config['RATE'],
                              input=True,
                              frames_per_buffer=config['CHUNK'],
                              input_device_index=config['input_device_idx'],
                              stream_callback=stream_callback)  # 指定输入设备索引

    # print("* recording")
    stream.start_stream()

    # # 等待外部停止信号
    # while stream.is_active() and config['record_status']:
    #     pass

    # print("* done recording")

    # 停止和关闭流
    if config['is_quit']:
        stream.stop_stream()
        stream.close()


def record_audio(headset_address, device_name, audio_queue):
    """录制蓝牙耳机音频并保存为 WAV 文件"""

    def callback(in_data, frame_count, time_info, status):
        audio_queue.put(in_data)
        return None, pyaudio.paContinue

    while not config['is_quit']:
        if config['record_status']:
            # 打开音频流
            stream = config['p'].open(format=config['FORMAT'],
                                      channels=config['CHANNELS'],
                                      rate=config['RATE'],
                                      input=True,
                                      frames_per_buffer=config['CHUNK'],
                                      input_device_index=config['input_device_idx'],
                                      stream_callback=callback)  # 指定输入设备索引

            print("* recording")
            stream.start_stream()

            # 等待外部停止信号
            while stream.is_active() and config['record_status']:
                pass

            print("* done recording")

            # 停止和关闭流
            stream.stop_stream()
            stream.close()
        else:
            time.sleep(1)


def save_audio_stream(audio_queue, save_filename):
    """从队列中获取音频帧并保存为 WAV 文件"""
    wf = wave.open(save_filename, 'wb')
    wf.setnchannels(config['CHANNELS'])
    wf.setsampwidth(config['p'].get_sample_size(config['FORMAT']))
    wf.setframerate(config['RATE'])
    config['recorded_frames'].clear()
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
            config['recorded_frames'].append(data)
            wf.writeframes(data)
        except queue.Empty:
            if not config['record_status']:  # 录音结束
                break

    wf.close()
    config['recorded_wav'] = b''.join(config['recorded_frames'])


def get_recorded_audio(audio_queue):
    """从缓存中获取录制的音频数据"""
    while not audio_queue.empty():
        config['recorded_frames'].append(audio_queue.get_nowait())
    return b''.join(config['recorded_frames'])


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


async def play_audio_async_pygame(matches_results, audio_folder):
    audio_file_path = config['mapping'].get(matches_results)

    if not config['is_playing']:
        config['is_playing'] = True

        def play_audio_thread():
            try:
                # print(f"\nPlaying audio: {audio_file_path}")
                pygame.mixer.init()
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
                config['is_playing'] = False

        config['play_thread'] = threading.Thread(target=play_audio_thread)
        config['play_thread'].start()


def save_image(image, subject):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S%f')
    file_path = os.path.join('images', f'image_{subject}_{timestamp}.jpg')

    with open(file_path, 'wb') as f:
        f.write(image)

    # print(f'Saved image to {file_path}')


def analyze(byte_array, pre_image_hist, pre_subject):
    np_array = np.frombuffer(byte_array, dtype=np.uint8)
    image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    query_histogram = calculate_color_histogram(image)
    if pre_image_hist is not None:
        sim_with_pre = compare_histograms(query_histogram, pre_image_hist)
        if sim_with_pre < 0.1:
            # print(pre_subject)
            return pre_subject, query_histogram

    results = []
    for i in range(len(config['data'])):
        similarity = compare_histograms(query_histogram, config['data'][i])
        results.append((i, similarity))
    results.sort(key=lambda x: x[1])

    id = int(results[0][0] / 30)
    # print(config['subjects'][id])
    return config['subjects'][id], query_histogram


async def run_bleak_client():
    async with BleakClient(config['DEVICE_ADDRESS']) as client:
        print("Connected")

        async def callback(sender, data):
            frame_number = int.from_bytes(data[:2], byteorder='little')

            if frame_number == 0xFFFF:
                # print("End of image data")
                subject, config['pre_image_hist'] = analyze(config['image_data'],
                                                            config['pre_image_hist'],
                                                            config['pre_subject'])
                if subject == config['pre_subject']:
                    if not config['is_asking']:
                        await play_audio_async_pygame(subject, config['audio_folder'])
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


def glass_stream():
    cap = cv2.VideoCapture(config['rtsp_url'])

    if not cap.isOpened():
        # print("Error: Could not open video stream.")
        exit()

    while not config['is_quit']:
        ret, frame = cap.read()
        if not ret:
            # print("Error: No more frames to read.")
            break

        # cv2.imshow('RTSP Stream', frame)
        config['photo'] = frame

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def analyze_glass(image, pre_image_hist):

    query_histogram = calculate_color_histogram(image)
    if pre_image_hist is not None:
        sim_with_pre = compare_histograms(query_histogram, pre_image_hist)
        # print(sim_with_pre)
        if sim_with_pre < 0.1:
            results = []
            for i in range(len(config['data'])):
                similarity = compare_histograms(query_histogram, config['data'][i])
                results.append((i, similarity))
            # 根据相似度排序检索结果
            results.sort(key=lambda x: x[1])
            # 相似度最高的类别
            # print(results[0][1])
            if results[0][1] < 0.35:
                id = int(results[0][0] / 30)
                # print(config['subjects'][id])
                return config['subjects'][id], query_histogram
    return None, query_histogram

async def run_glass_retrieval():
    last_saved_time = time.time()

    while True:
        current_time = time.time()
        if current_time - last_saved_time >= config['interval'] and config['photo'] is not None:
            # 保存图像
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S%f')
            file_path = os.path.join('images', f'image_{timestamp}.jpg')
            #cv2.imwrite(file_path,global_vars['photo'])

            subject, config['pre_image_hist'] = analyze_glass(config['photo'], config['pre_image_hist'])
            if subject is not None:
                await play_audio_async_pygame(subject, config['audio_folder'])
            last_saved_time = current_time  # 更新上次保存时间

def glass_retrieval():
    asyncio.run(run_glass_retrieval())

def bleak_client():
    asyncio.run(run_bleak_client())


def main():
    if not os.path.exists('images'):
        os.makedirs('images')

    # 在单独的线程中运行 BleakClient
    threading.Thread(target=bleak_client).start()

    # 简单的 pygame 事件循环
    running = True
    while not config['is_quit']:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

    pygame.quit()


def handle_user_input(record_queue, play_queue):
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
        # elif user_input.lower() == "start record":
        #     config['is_asking'] = True
        #     # if not config['record_status']:  # 只有在未录音状态下才能开始录音
        #     config['record_status'] = True
        #     record_count = record_count + 1
        #     save_thread = threading.Thread(target=save_audio_stream, args=(record_queue, f"./audio/ask.wav"))
        #     save_thread.start()
        # elif user_input.lower() == "stop record":
        #     config['record_status'] = False
        #     save_thread.join()
        #     save_thread = None
        #     print(f"Recorded audio length: {len(config['recorded_wav'])} bytes")
        #
        #     # 停止录音后，提示用户输入要播放的音频文件
        #     chat_with_llm()
        #     filename = "./audio/answer.wav"
        #     # filename = "test.wav"
        #     audio_thread = threading.Thread(target=play_audio_pygame,
        #                                     args=(headset_addr, config['target_device'], filename, play_queue))
        #     audio_thread.start()
        #     audio_thread.join()  # 等待播放线程结束
        #     audio_thread = None
        #     print("out of audio play")
        #     config['is_asking'] = False
        # elif user_input.lower() == "stop play":
        #     play_queue.put(None)  # 向队列中放入 None 表示停止播放


def vad():
    # Start audio with VAD
    vad_audio = VADAudio(aggressiveness=config['webRTC_aggressiveness'],
                         device=config['input_device_idx'],
                         input_rate=config['RATE'])

    print("Listening (ctrl-C to exit)...")
    frames = vad_audio.vad_collector()

    # load silero VAD
    os.environ['TORCH_HOME'] = 'D://torch_models//cache'
    torchaudio.set_audio_backend("soundfile")
    model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                  model=config['silaro_model_name'],
                                  force_reload=config['reload'])
    (get_speech_ts, _, _, _, _) = utils

    # Stream from microphone to DeepSpeech using VAD
    spinner = None
    if not config['nospinner']:
        spinner = Halo(spinner='line')
    wav_data = bytearray()
    for frame in frames:
        if frame is not None:
            if spinner: spinner.start()
            wav_data.extend(frame)

        else:
            if spinner: spinner.stop()
            print("webRTC has detected a possible speech")

            newsound = np.frombuffer(wav_data, np.int16)
            audio_float32 = Int2Float(newsound)
            print(wav_data.__class__)
            time_stamps = get_speech_ts(audio_float32, model)

            if len(time_stamps) > 0:
                print("silero VAD has detected a possible speech")
                start = int(time_stamps[0]['start'])
                end = int(time_stamps[0]['end'])
                audio = newsound.astype(np.float32) / 32768.0
                audio_clip = audio[start:end]

                audio_clip = audio_clip * 5

                audio_clip_int16 = (audio_clip * 32768.0).astype(np.int16)
                audio_bytes = audio_clip_int16.tobytes()
                output_wav_path = './audio/audio_clip.wav'
                with contextlib.closing(wave.open(output_wav_path, 'wb')) as wav_file:
                    wav_file.setnchannels(config['CHANNELS'])  # 单声道
                    wav_file.setsampwidth(2)  # 2 字节样本宽度
                    wav_file.setframerate(config['RATE'])
                    wav_file.writeframes(audio_bytes)
            else:
                print("silero VAD has detected a noise")
            print()
            wav_data = bytearray()


if __name__ == "__main__":
    if not os.path.exists('images'):
        os.makedirs('images')

    headset_addr = find_bluetooth_headset(config['target_device'])
    if headset_addr:
        print("addr", headset_addr)
        record_queue = queue.Queue()  # 创建音频帧队列
        play_queue = queue.Queue()  # 创建音频帧队列

        print("Bluetooth headset found, starting function...")

        # 启动蓝牙线程
        # bleak_thread = threading.Thread(target=bleak_client)
        # bleak_thread.start()

        # 启动眼镜视频流
        # glass_stream_thread = threading.Thread(target=glass_stream)
        # glass_stream_thread.start()

        #启动眼镜图像分析线程
        # glass_retrieval_thread = threading.Thread(target=glass_retrieval)
        # glass_retrieval_thread.start()

        # 启动录音线程
        record_thread = threading.Thread(target=vad)
        record_thread.start()

        # 启动用户输入处理线程
        input_thread = threading.Thread(target=handle_user_input, args=(record_queue, play_queue))
        input_thread.start()

        # 简单的 pygame 事件循环
        # running = True
        # while not config['is_quit']:
        #     for event in pygame.event.get():
        #         if event.type == pygame.QUIT:
        #             running = False

        # pygame.quit()

        # 等待线程结束
        record_thread.join()
        input_thread.join()
        bleak_thread.join()
        config['p'].terminate()

        print("Audio saved to record.wav")
    else:
        print("Failed to find Bluetooth headset.")
