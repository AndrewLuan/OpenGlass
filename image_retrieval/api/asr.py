import numpy as np
import json
import wave
import base64
import websocket
import uuid
import hashlib
import time
import os
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv('../env/.env'))


# asr_apiKey = os.getenv("ASR_API_KEY")
# asr_secretKey = os.getenv("ASR_SECRET_KEY")


# 此处websocket库为websocket-client


# 查询音频文件采样率，通道数等信息
def wav_to_info(filename):
    with wave.open(filename, 'r') as wav_file:
        # 获取音频文件的参数
        params = wav_file.getparams()
        nchannels, sampwidth, framerate, nframes = params[:4]

        print(f"通道数: {nchannels}")
        print(f"样本宽度: {sampwidth}")
        print(f"采样率: {framerate}")
        print(f"帧数: {nframes}")


# WAV文件转换成base64编码
def wav_to_base64(filename):
    with wave.open(filename, 'rb') as wav_file:
        wav_data = wav_file.readframes(wav_file.getnframes())
        base64_data = base64.b64encode(wav_data)
        return base64_data.decode('utf-8')


# wav转pcm再转换成base64编码
def wav2pcm(wavfile, data_type=np.int16):
    f = open(wavfile, "rb")
    f.seek(0)
    f.read(44)
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
    # 创建SHA-256对象
    sha256 = hashlib.sha256()

    # 更新哈希对象的内容
    sha256.update(message.encode('utf-8'))

    # 计算哈希值
    hash_value = sha256.hexdigest()

    return hash_value


# 获取时间戳和填写apiKey、secretKey
timestamp = str(int(time.time()))
apiKey = os.getenv("ASR_API_KEY")
secretKey = os.getenv("ASR_SECRET_KEY")

traceId = str(uuid.uuid4())
apisign = sha256_hash(apiKey + '-' + secretKey +
                      '-' + traceId + '-' + timestamp)
ws_url = ("wss://150.223.245.42/csrobot/cschannels/openapi/ws/asr?apiKey="
          + apiKey + "&appSign=" + apisign + "&traceId=" + traceId + "&timestamp=" + timestamp)
#
# ws_url="ws://10.127.23.49:18080/v1/asr"
print(ws_url)
# time.sleep(600)
# 创建一个websocket连接
ws = websocket.create_connection(ws_url)
# 从服务器接收消息
response = ws.recv()
print("Received message:", json.loads(response)["data"])
# 发送开始识别消息到服务器
# req_id实际使用时需要替换为随机值
ws.send(json.dumps({"req_id": str(uuid.uuid4()), "rec_status": 0}))
print("开始识别Message sent")
# 从服务器接收消息
response2 = ws.recv()
print("Received message:", json.loads(response2)["message"])
# 发送语音文件到服务器
# demo使用固定文件，实际使用需要传输实时pcm
wav_path = '../../sound/sample-3s.wav'
wav_to_info(wav_path)
# 此处示例了3种方法，wav2pcm是wav文件转成pcm再转base64,wav_to_base64是wav文件直接转base64，pcm_to_base64是pcm文件直接转base64
base64_string = wav2pcm(wav_path)
# base64_string = wav_to_base64(wav_path)
# pcm_path = ''
# base64_string = pcm_to_base64(wav_path)
# 此处使用了默认参数，如果音频文件非单通道或非16000的采样率，识别效果会不理想，可以使用文件中wav_to_info方法查出wav音频的文件信息，
# 非单通道需要转换为单通道，非16000的采样率可以在首包中加上采样率参数
voice = json.dumps({"audio_stream": base64_string, "rec_status": 1})
ws.send(voice)
print("开始识别Message sent")
ws.send(json.dumps({"rec_status": 2}))
print("结束调用Message sent")
try:
    voices = []
    # 循环接收流式数据
    while True:
        # 接收数据
        result = ws.recv()
        # print(1)
        if result:
            # print(2)
            # 返回的识别信息用后返回text中的信息覆盖之前返回的
            voice = json.loads(result)["data"]["results"][0]["text"]
            print(voice)
        # res_status=3代表一句话识别结束
        if json.loads(result)["res_status"] == 3:
            # print(3)
            voice = json.loads(result)["data"]["results"][0]["text"]
            voices.append(voice)
            print("单句识别结果：" + voice)
        # res_status=4代表整个语音识别结束
        if json.loads(result)["res_status"] == 4:
            finally_voice = ""
            for i in voices:
                finally_voice = finally_voice + i
                print("最终结果：" + finally_voice)
            break
except Exception as e:
    print(e)
finally:
    # 发送结束调用传输到服务器
    # response3 = ws.recv()
    # print("Received message:", json.loads(response3)["message"])
    # 关闭连接
    ws.close()
    print("关闭成功")
