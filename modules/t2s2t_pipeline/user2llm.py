from openai import OpenAI
 

 
def chat(client, query, history):
    history.append({
        "role": "user", 
        "content": query
    })
    completion = client.chat.completions.create(
        model="moonshot-v1-8k",
        messages=history,
        temperature=0.3,
    )
    result = completion.choices[0].message.content
    history.append({
        "role": "assistant",
        "content": result
    })
    return result


# print(chat("月球呢？", history))
if __name__ == "__mian__":
    client = OpenAI(
    api_key = "sk-uSS6OrqyCA4XSdUXiPDQyaX5M6wZhwNG6etQx6rUja6IMm47",
    base_url = "https://api.moonshot.cn/v1",
)

    history = [
    {"role": "system", "content": "你是问答任务的助手。使用以下检索到的上下文和对话历史记录来继续对话。\
        如果你不知道答案，就说你没有找到相关信息或者你不知道。最多使用三个句子并保持答案简洁。\
        如果消息不需要上下文，它将是空的，所以随意回答问题。"}
]
    print(chat("我刚来到上海，你能为我推荐几家饭店吗", history))