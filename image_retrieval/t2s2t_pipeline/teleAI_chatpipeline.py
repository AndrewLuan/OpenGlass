from teleAI_s2t import s2t
from teleAI_llm import llmchat
from teleAI_t2s import t2s
from user2llm import chat
from tts_http_demo import tts_bd
from openai import OpenAI

if __name__ == '__main__':
    input_wav = 'tmp.wav'
    s2t_api_key = "1CD9A7F5E5D54F39878394B8B841D9DC"
    s2t_scr_key = "48E7829693D3472BAD52B292F5C5499B"
    user_input = s2t(s2t_api_key, s2t_scr_key, input_wav)

    # llm_api_key = "599BA3BC3B87490283F271FB6EA2F4A4"
    # llm_scr_key = "E49B9F4CD2EB47448480B876BE087E01"
    # llm_url = "https://150.223.245.42/csrobot/cschannels/openapi/chat/dialog?apiKey=599BA3BC3B87490283F271FB6EA2F4A4"
    # llm_output = llmchat(llm_api_key, llm_scr_key, llm_url, user_input)
    # llm_output = llm_output.replace("\n", "")
    # llm_output = llm_output.replace("。", "，")
    # t2s_api_key = "81EECFFE87994972BAA0B352F43FFB7E"
    # t2s_scr_key = "14C330D5A9FE4B58940F53DD15EFC379"
    # output_wav = t2s(t2s_api_key, t2s_scr_key, llm_output)
    KIMI_API_KEY = "sk-uSS6OrqyCA4XSdUXiPDQyaX5M6wZhwNG6etQx6rUja6IMm47"
    history = []
    client = OpenAI(
        api_key = KIMI_API_KEY,
        base_url = "https://api.moonshot.cn/v1",
    )
    llm_output = chat(client, user_input, history)
    output_wav = tts_bd(llm_output)
    print(output_wav)
