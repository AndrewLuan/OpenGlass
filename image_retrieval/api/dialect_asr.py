import time
import hashlib
import uuid
import websocket
import base64
import wave
import json
import numpy as np
import os
from dotenv import load_dotenv, find_dotenv

# 加载环境变量
_ = load_dotenv(find_dotenv('../env/.env'))


# 查询音频文件采样率，通道数等信息
def wav_to_info(filename):
    with wave.open(filename, 'r') as wav_file:
        params = wav_file.getparams()
        nchannels, sampwidth, framerate, nframes = params[:4]
        return {
            "channels": nchannels,
            "sample_width": sampwidth,
            "frame_rate": framerate,
            "frames": nframes
        }


# WAV文件转换成base64编码
def wav_to_base64(filename):
    with wave.open(filename, 'rb') as wav_file:
        wav_data = wav_file.readframes(wav_file.getnframes())
        base64_data = base64.b64encode(wav_data)
        return base64_data.decode('utf-8')


# wav转pcm再转换成base64编码
def wav2pcm(wavfile, data_type=np.int16):
    with open(wavfile, "rb") as f:
        f.seek(0)
        f.read(44)  # 跳过WAV文件头
        data = np.fromfile(f, dtype=data_type)
        base64_data = base64.b64encode(data)
        return base64_data.decode('utf-8')


# PCM文件转换成base64编码
def pcm_to_base64(pcm_file_path):
    with open(pcm_file_path, 'rb') as pcm_file:
        pcm_data = pcm_file.read()
        base64_data = base64.b64encode(pcm_data)
        return base64_data.decode('utf-8')


# 签名需要SHA256加密
def sha256_hash(message):
    sha256 = hashlib.sha256()
    sha256.update(message.encode('utf-8'))
    return sha256.hexdigest()


# 获取时间戳和api签名
def get_api_signature(apiKey, secretKey):
    timestamp = str(int(time.time()))
    traceId = str(uuid.uuid4())
    apisign = sha256_hash(apiKey + '-' + secretKey +
                          '-' + traceId + '-' + timestamp)
    return timestamp, traceId, apisign


# 创建WebSocket连接
def create_ws_connection(ws_url):
    ws = websocket.create_connection(ws_url)
    response = ws.recv()
    print("Received message:", json.loads(response)["data"])
    return ws


# 发送识别请求
def send_recognition_request(ws, req_id=None):
    if req_id is None:
        req_id = str(uuid.uuid4())
    ws.send(json.dumps({"req_id": req_id, "rec_status": 0}))
    print("开始识别Message sent")
    response = ws.recv()
    print("Received message:", json.loads(response)["message"])


# 发送语音数据
def send_audio_data(ws, audio_base64):
    voice = json.dumps({"audio_stream": audio_base64, "rec_status": 1})
    ws.send(voice)
    print("语音数据已发送")
    ws.send(json.dumps({"rec_status": 2}))
    print("结束调用Message sent")


# 处理语音识别结果
def process_recognition_result(ws):
    voices = []
    try:
        while True:
            result = ws.recv()
            if result:
                voice = json.loads(result)["data"]["results"][0]["text"]
                print(voice)
                if json.loads(result)["res_status"] == 3:
                    voices.append(voice)
                    print("单句识别结果：" + voice)
                if json.loads(result)["res_status"] == 4:
                    finally_voice = "".join(voices)
                    print("最终结果：" + finally_voice)
                    return finally_voice
    except Exception as e:
        print(e)
    finally:
        ws.close()
        print("连接已关闭")


# 主函数，整合上述步骤
def recognize_speech(wav_path):
    apiKey = os.getenv("DIALECT_RECOGNITION_API_KEY")
    secretKey = os.getenv("DIALECT_RECOGNITION_SECRET_KEY")

    timestamp, traceId, apisign = get_api_signature(apiKey, secretKey)

    ws_url = (
        "wss://150.223.245.42/csrobot/cschannels/openapi/ws/asr?apiKey="
        + apiKey + "&appSign=" + apisign + "&traceId=" +
        traceId + "&timestamp=" + timestamp
    )

    ws = create_ws_connection(ws_url)

    send_recognition_request(ws)

    audio_info = wav_to_info(wav_path)
    print(audio_info)

    # 示例：使用wav2pcm方法将wav文件转为base64编码
    base64_string = wav2pcm(wav_path)

    send_audio_data(ws, base64_string)

    final_result = process_recognition_result(ws)
    return final_result


if __name__ == '__main__':
    # 示例调用
    wav_path = '周一周二周三.wav'
    result = recognize_speech(wav_path)
    print("识别结果：", result)
