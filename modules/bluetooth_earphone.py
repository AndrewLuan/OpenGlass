import pyaudio  # 替换 sounddevice
import wave
import bluetooth  # 用于蓝牙扫描和连接
import pygame
import time
import numpy as np
import queue
import threading
import asyncio


def find_bluetooth_headset(device_name):
    """搜索蓝牙耳机并返回设备地址"""
    nearby_devices = bluetooth.discover_devices(lookup_names=True)
    print(nearby_devices)
    for addr, name in nearby_devices:
        if device_name in name:  # 根据实际耳机名称修改
            return addr
    return None


def find_input_device_idx(device_name, p_pyaudio):
    # 查找蓝牙耳机麦克风输入设备索引
    for i in range(p_pyaudio.get_device_count()):
        dev_info = p_pyaudio.get_device_info_by_index(i)
        if device_name in str(dev_info['name']) and dev_info['maxInputChannels'] > 0:
            return i
    else:
        raise Exception("Bluetooth headset microphone not found")


def find_output_device_idx(device_name, p_pyaudio):
    # 查找蓝牙耳机麦克风输出设备索引
    for i in range(p_pyaudio.get_device_count()):
        dev_info = p_pyaudio.get_device_info_by_index(i)
        if device_name in str(dev_info['name']) and dev_info['maxOutputChannels'] > 0:
            return i
    else:
        raise Exception("Bluetooth headset output not found")


def play_audio(config, filename, audio_queue):
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


def play_audio_pygame(filename, audio_queue):
    # Pygame 初始化，包括音频模块
    pygame.mixer.init()

    # 1. 加载音频文件
    sound = pygame.mixer.Sound(filename)

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


def record_audio(config, audio_queue):
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


def save_audio_stream(config, audio_queue, save_filename):
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


def get_recorded_audio(config, audio_queue):
    """从缓存中获取录制的音频数据"""
    while not audio_queue.empty():
        config['recorded_frames'].append(audio_queue.get_nowait())
    return b''.join(config['recorded_frames'])


async def play_audio_async_pygame(config, matches_results, audio_folder):
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

