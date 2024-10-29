# 基本介绍

本文件夹为用户语音输入-语音转文字-llm文字输入-llm文字输出-文字转语音的pipeline，本文将介绍配置该pipeline的环境。
**注意：目前所有使用的api均为免费额度，具有限额或速度等的限制**

# 语音转文字 (speech2text)

所用模型为DeepGram框架下的nova-v2：https://developers.deepgram.com/docs/getting-started-with-pre-recorded-audio

Install SDK:

        # Install the Deepgram Python SDK
        # https://github.com/deepgram/deepgram-python-sdk

        pip install deepgram-sdk


Add Dependencies


        # Install python-dotenv to protect your API key

        pip install python-dotenv


可以配置具体的语言，也可检测，该模型中文输出格式需要额外处理，效果不好：


        options = PrerecordedOptions(
                model="nova-2",
                smart_format=True,
                detect_language=True # 可换为具体的语言，如'zh', 'en'
            )


# llm交互 (user2llm)

使用的llm为kimi：https://kimi.moonshot.cn/，
其API兼容OpenAI：https://platform.moonshot.cn/docs/，
需要保证openai>=1.0，python>=3.8


        # pip install openai

        from openai import OpenAI
 
        client = OpenAI(
            api_key="MOONSHOT_API_KEY", # <--在这里将 MOONSHOT_API_KEY 替换为API Key
            base_url="https://api.moonshot.cn/v1", # <-- 将 base_url 从 https://api.openai.com/v1 替换为 https://api.moonshot.cn/v1
        )


# 文字转语音 (text2speech)

目前仅支持英文，同样使用DeepGram框架，可使用同一个api：https://developers.deepgram.com/docs/text-to-speech

# 中文文字转语音（tts_http_demo）

使用火山引擎进行合成，支持中文

# teleAI语音转文字（teleAI_s2t）

在中文环境下，该效果优于dg框架

# 中文单轮对话pipeline（teleAI_chatpipeline）

使用teleAI进行语音转文字，llm为kimi，使用火山引擎进行文字转语音
