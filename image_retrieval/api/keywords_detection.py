import time
import hashlib
import uuid
import websocket
import base64
import wave
import json
import os
from dotenv import load_dotenv, find_dotenv

# 加载环境变量
_ = load_dotenv(find_dotenv('../env/.env'))

# WAV文件转换成base64编码


def wav_to_base64(filename):
    with wave.open(filename, 'rb') as wav_file:
        wav_data = wav_file.readframes(wav_file.getnframes())
        base64_data = base64.b64encode(wav_data)
        return base64_data.decode('utf-8')


# 签名需要SHA256加密
def sha256_hash(message):
    sha256 = hashlib.sha256()
    sha256.update(message.encode('utf-8'))
    return sha256.hexdigest()


# 获取API签名和相关参数
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
def send_keyword_detection_request(ws, keywords):
    req_id = "keyWord"
    ws.send(json.dumps(
        {"req_id": req_id, "det_status": 0, "kw_list": keywords}))
    print("开始识别Message sent")
    response = ws.recv()
    print("Received message:", json.loads(response)["status_msg"])


# 发送语音数据
def send_audio_data(ws, wav_path):
    wav_base64_string = wav_to_base64(wav_path)
    voice = json.dumps({"audio_stream": wav_base64_string,
                       "det_status": 1, "is_end": False})
    ws.send(voice)
    print("语音编码Message sent")


# 处理关键词检测结果
def process_detection_result(ws):
    try:
        results = []
        while True:
            result = ws.recv()
            if result:
                result = json.loads(result)
                num = 1
                for i in result["result"]:
                    detection_result = (
                        f"{num}.识别到'{i['kw']}'关键词，起始于语音第{i['begin_time']}毫秒处，"
                        f"结束于{i['end_time']}毫秒处，可信度"
                    )
                    results.append(detection_result)
                    num += 1
                if results:
                    print("\n".join(results))
                break
            else:
                print("未识别到关键词")
                break
    except Exception as e:
        print(e)
    finally:
        ws.send(json.dumps({"det_status": 2}))
        print("结束调用Message sent")
        response3 = ws.recv()
        print("Received message:", json.loads(response3)["status_msg"])
        ws.close()
        print("连接已关闭")


# 主函数，整合所有步骤
def detect_keywords_in_audio(wav_path, keywords):
    apiKey = os.getenv("STREAMING_KEYWORD_DETECTION_API_KEY")
    secretKey = os.getenv("STREAMING_KEYWORD_DETECTION_SECRET_KEY")

    timestamp, traceId, apisign = get_api_signature(apiKey, secretKey)

    ws_url = (
        "wss://150.223.245.42/csrobot/cschannels/openapi/ws/voiceKeyword?apiKey="
        + apiKey + "&appSign=" + apisign + "&traceId=" +
        traceId + "&timestamp=" + timestamp
    )

    ws = create_ws_connection(ws_url)

    send_keyword_detection_request(ws, keywords)

    send_audio_data(ws, wav_path)

    process_detection_result(ws)


if __name__ == '__main__':
    wav_path = 'Demo voice.wav'
    keywords = ["下雨", "伞"]
    detect_keywords_in_audio(wav_path, keywords)
