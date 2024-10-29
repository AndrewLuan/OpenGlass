from user2llm import chat
import os
from dotenv import load_dotenv
from openai import OpenAI
from text2speech import text2speech
from speech2text import speech2text

from deepgram import (
    DeepgramClient,
    PrerecordedOptions,
    FileSource,
)

if __name__ == "__main__":
    load_dotenv()

    # Path to the audio file
    AUDIO_FILE = "input_test_en.wav"
    OUTPUT_FILE = "output.wav"

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
