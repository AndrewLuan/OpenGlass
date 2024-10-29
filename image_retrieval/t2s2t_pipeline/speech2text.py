# main.py (python example)
from user2llm import chat
import os
from dotenv import load_dotenv
from openai import OpenAI
from text2speech import text2speech

from deepgram import (
    DeepgramClient,
    PrerecordedOptions,
    FileSource,
)

def speech2text(file_path, dg):
    with open(file_path, "rb") as file:
        buffer_data = file.read()

    payload: FileSource = {
        "buffer": buffer_data,
    }

        #STEP 2: Configure Deepgram options for audio analysis
    options = PrerecordedOptions(
        model="nova-2",
        smart_format=True,
        detect_language='en'
    )

    # STEP 3: Call the transcribe_file method with the text payload and options
    response = dg.listen.rest.v("1").transcribe_file(payload, options)

    # STEP 4: Print the response
    # print(response.to_json(indent=4))
    # Assuming `response` is the response object from the Deepgram API
    transcript = response['results']['channels'][0]['alternatives'][0]['transcript']
    print(transcript)
    # result = transcript.replace(" ", "")
    # print(result)
    return transcript

if __name__ == "__main__":
    load_dotenv()

# Path to the audio file
    AUDIO_FILE = "backend/testing/test_en.wav"
    OUTPUT_FILE = "backend/testing/output.wav"

    DG_API_KEY = '1fda42ddec0455802e88971738e46aa9fc32e3c7'

    client = OpenAI(
        api_key = "sk-uSS6OrqyCA4XSdUXiPDQyaX5M6wZhwNG6etQx6rUja6IMm47",
        base_url = "https://api.moonshot.cn/v1",
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



# def main():
#     try:
#         # STEP 1 Create a Deepgram client using the API key
#         deepgram = DeepgramClient(DG_API_KEY)

#         with open(AUDIO_FILE, "rb") as file:
#             buffer_data = file.read()

#         payload: FileSource = {
#             "buffer": buffer_data,
#         }

#         #STEP 2: Configure Deepgram options for audio analysis
#         options = PrerecordedOptions(
#             model="nova-2",
#             smart_format=True,
#             detect_language='zh'
#         )

#         # STEP 3: Call the transcribe_file method with the text payload and options
#         response = deepgram.listen.rest.v("1").transcribe_file(payload, options)

#         # STEP 4: Print the response
#         # print(response.to_json(indent=4))
#         # Assuming `response` is the response object from the Deepgram API
#         transcript = response['results']['channels'][0]['alternatives'][0]['transcript']
#         print(transcript)
#         result = transcript.replace(" ", "")
#         print(result)
#     except Exception as e:
#         print(f"Exception: {e}")



