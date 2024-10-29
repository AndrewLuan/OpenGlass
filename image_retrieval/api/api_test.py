import uuid
import time
import requests
import hashlib
import json
import pdb

import os
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv('../env/.env'))
# 替换为您的secretKey和应用Id、应用key
apiKey = os.getenv("CHAT_API_KEY")
secretKey = os.getenv("CHAT_SECRET_KEY")

# 请求地址，请向研发老师索取最新地址
url = f'''https://150.223.245.42/csrobot/cschannels/openapi/chat/dialog?apiKey={apiKey}'''


# 输入内容
content_list = ["请写出一篇以人工智能为题的文章，不少于500字，请按照markdown格式编写。",
                "请介绍一下原神",
                "请问你是谁？",
                "你是人工智能吗？",
                "请介绍一下中国画"]

for content in content_list:
    # 创建消息列表
    traceId = uuid.uuid4()
    timestamp = int(time.time())
    start_time = time.time()
    messages = [{"role": "user", "content": content}]
    # 使用字符串构建器生成签名字符串
    string_builder = ""
    for message in messages:
        string_builder += message["content"] + "-"
    # 待加密内容
    sign_str = f"{apiKey}-{secretKey}-{traceId}-{timestamp}"
    # 签名
    sign = hashlib.sha256(sign_str.encode('utf-8')).hexdigest()
    # 构造请求头
    headers = {
        "App-Sign": sign,
        "Content-Type": "application/json;UTF-8"
    }
    # 构造请求体
    data = {
        "traceId": str(traceId),
        "timestamp": timestamp,
        "stream": False,
        "messages": messages
    }
    print(traceId)
    print(timestamp)
    print(sign)
    # 发送POST请求
    response = requests.post(url, headers=headers, json=data)
    # pdb.set_trace()
    # # 打印原始响应内容
    # 打印原始响应内容（字节字符串）
    # print("Raw response content:", response.content)
    # 尝试将响应内容解码为字符串
    try:
        decoded_content = response.content.decode('utf-8')  # 根据实际编码选择合适的解码格式
        end_time = time.time() - start_time
        print("Decoded response content:", decoded_content)
    except UnicodeDecodeError as e:
        print("Failed to decode response content:", e)
    # # 处理SSE响应
    # if response.status_code == 200:
    #     current_event = ""
    #     for line in response.iter_lines():
    #         print(line)
    # else:
    #     print(f"请求失败，状态码：{response.status_code}")
    # print(current_event)
