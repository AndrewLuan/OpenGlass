import time
import hashlib
import uuid
import websocket
import base64
import wave
import json
import os
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv('../env/.env'))

# base64编码转换成WAV文件


def wav_to_base64(data):
    # 解码Base64字符串
    audio_data = base64.b64decode(data)
    # 打开WAV文件写入器
    filename = 'voice.wav'
    with wave.open(filename, 'wb') as wf:
        # 设置WAV文件的参数，这里需要知道音频的具体参数（采样率、通道数、位深度等）
        nchannels = 1  # 单声道为1，立体声为2
        sampwidth = 2  # 每个样本的字节数，通常为1（8位）或2（16位）
        framerate = 16000  # 采样率，通常为44100Hz
        nframes = len(audio_data) // (nchannels * sampwidth)  # 计算总帧数
        comptype = "NONE"  # 压缩类型，如果是未压缩则为"NONE"
        compname = "not compressed"  # 压缩名称
        # 设置WAV文件头信息
        wf.setparams((nchannels, sampwidth, framerate,
                     nframes, comptype, compname))
        # 写入数据
        wf.writeframes(audio_data)
    print('语音合成文件以生成：' + filename)


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
apiKey = os.getenv("TTS_API_KEY")
secretKey = os.getenv("TTS_SECRET_KEY")

# traceId随机生成
traceId = str(uuid.uuid4())
apisign = sha256_hash(apiKey + '-' + secretKey +
                      '-' + traceId + '-' + timestamp)
ws_url = ("wss://150.223.245.42/csrobot/cschannels/openapi/ws/tts?apiKey="
          + apiKey + "&appSign=" + apisign + "&traceId=" + traceId + "&timestamp=" + timestamp)
print(ws_url)
# time.sleep(600)
# 创建一个websocket连接
ws = websocket.create_connection(ws_url)
# 从服务器接收消息
response = ws.recv()
print("Received message:", json.loads(response)["data"])
# 发送开始识别消息到服务器
# req_id实际使用时需要替换为随机值，text为需要合成语音的文字
ws.send(json.dumps({"req_id": str(uuid.uuid4()), "text": "下雨了,还好我带了伞。"}))
print("需要语音合成的文字Message sent")
# 从服务器接收消息
response2 = ws.recv()
print("Received message:", json.loads(response2)["status_msg"])
# 发送语音文件到服务器
try:
    # 循环接收流式数据
    voice = ""
    while True:
        # 接收数据
        result = ws.recv()
        if result:
            # 返回的⾳频数据需要拼接起来再转为语音
            voice = voice + json.loads(result)["result"]["audio"]
        # is_end=True代表合成结束
        if json.loads(result)["result"]["is_end"]:
            # 将拼接好的base64码转换为wav语音文件
            wav_to_base64(voice)
            break
except Exception as e:
    print(e)
finally:
    # 关闭连接
    ws.close()
    print("关闭成功")
