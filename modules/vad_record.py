import os
import wave

import numpy as np

import torch
import torchaudio

from .vad import Audio, VADAudio, Int2Float
from halo import Halo
import contextlib
import threading
from .chat_llm import chat_with_llm
from .bluetooth_earphone import play_audio_pygame


def vad_record(config, play_queue):
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
                config['is_asking'] = True
                start = int(time_stamps[0]['start'])
                end = int(time_stamps[0]['end'])
                audio = newsound.astype(np.float32) / 32768.0
                audio_clip = audio[start:end]

                audio_clip = audio_clip * 5

                audio_clip_int16 = (audio_clip * 32768.0).astype(np.int16)
                audio_bytes = audio_clip_int16.tobytes()
                output_wav_path = './audio/ask.wav'
                with contextlib.closing(wave.open(output_wav_path, 'wb')) as wav_file:
                    wav_file.setnchannels(config['CHANNELS'])  # 单声道
                    wav_file.setsampwidth(2)  # 2 字节样本宽度
                    wav_file.setframerate(config['RATE'])
                    wav_file.writeframes(audio_bytes)

                # 停止录音后，提示用户输入要播放的音频文件
                # chat_with_llm()
                # filename = "./audio/answer.wav"
                # # filename = "test.wav"
                # audio_thread = threading.Thread(target=play_audio_pygame,
                #                                 args=(
                #                                     config['target_device'], filename, play_queue))
                # audio_thread.start()
                # audio_thread.join()  # 等待播放线程结束
                # audio_thread = None
                # print("out of audio play")
                config['is_asking'] = False
            else:
                print("silero VAD has detected a noise")
            print()
            wav_data = bytearray()
