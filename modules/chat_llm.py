from .t2s2t_pipeline.user2llm import chat
from dotenv import load_dotenv
from openai import OpenAI
from .t2s2t_pipeline.text2speech import text2speech
from .t2s2t_pipeline.speech2text import speech2text

from deepgram import (
    DeepgramClient,
    PrerecordedOptions,
    FileSource,
)


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

