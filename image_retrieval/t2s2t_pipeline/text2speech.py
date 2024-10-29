import os
from dotenv import load_dotenv

from deepgram import (
    DeepgramClient,
    SpeakOptions,
)

load_dotenv()



def text2speech(output_path, dg, text_options):
    options = SpeakOptions(
        model="aura-asteria-en",
        encoding="linear16",
        container="wav"
    )
    response = dg.speak.v("1").save(output_path, text_options, options)
    return True

def main():
    try:
        # STEP 1: Create a Deepgram client using the API key from environment variables
        deepgram = DeepgramClient(api_key=DG_API_KEY)

        # STEP 2: Configure the options (such as model choice, audio configuration, etc.)
        options = SpeakOptions(
            model="aura-asteria-en",
            encoding="linear16",
            container="wav"
        )

        # STEP 3: Call the save method on the speak property
        response = deepgram.speak.v("1").save(filename, SPEAK_OPTIONS, options)
        print(response.to_json(indent=4))

    except Exception as e:
        print(f"Exception: {e}")


if __name__ == "__main__":
    SPEAK_OPTIONS = {"text": "Hello, how can I help you today?"}
    filename = "backend/testing/output.wav"
    DG_API_KEY = '1fda42ddec0455802e88971738e46aa9fc32e3c7'
    main()
