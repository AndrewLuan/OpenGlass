#coding=utf-8

'''
requires Python 3.6 or later
pip install requests
'''
import base64
import json
import uuid
import requests

# 填写平台申请的appid, access_token以及cluster
def tts_bd(content):
    appid = "3564684675"
    access_token= "xRpKVn2p_zU2wyksWuFNAZ-crQSUus1n"
    cluster = "volcano_tts"

    voice_type = "BV007"
    host = "openspeech.bytedance.com"
    api_url = f"https://{host}/api/v1/tts"

    header = {"Authorization": f"Bearer;{access_token}"}

    request_json = {
        "app": {
            "appid": appid,
            "token": access_token,
            "cluster": cluster
        },
        "user": {
            "uid": "388808087185088"
        },
        "audio": {
            "voice_type": voice_type,
            "encoding": "wav",
            "speed_ratio": 1.0,
            "volume_ratio": 1.0,
            "pitch_ratio": 1.0,
        },
        "request": {
            "reqid": str(uuid.uuid4()),
            "text": content,
            "text_type": "plain",
            "operation": "query",
            "with_frontend": 1,
            "frontend_type": "unitTson"

        }
    }
    file_path = ""
    try:
        resp = requests.post(api_url, json.dumps(request_json), headers=header)
        # print(f"resp body: \n{resp.json()}")
        
        if "data" in resp.json():
            file_path = "test_submit.wav"
            data = resp.json()["data"]
            file_to_save = open(file_path, "wb")
            file_to_save.write(base64.b64decode(data))
    except Exception as e:
        e.with_traceback()
    finally:
        return file_path

if __name__ == '__main__':
    try:
        resp = requests.post(api_url, json.dumps(request_json), headers=header)
        # print(f"resp body: \n{resp.json()}")
        if "data" in resp.json():
            data = resp.json()["data"]
            file_to_save = open("test_submit.wav", "wb")
            file_to_save.write(base64.b64decode(data))
    except Exception as e:
        e.with_traceback()
