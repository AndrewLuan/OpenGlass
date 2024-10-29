from openai import OpenAI


system_messages  = [
        {"role": "system", "content": "你是生活语音助手。\
        如果你不知道答案，就说你没有找到相关信息或者你不知道。\
        如果消息不需要上下文，它将是空的，所以随意回答问题。请注意：回答中不要出现拼音和特殊符号如*&￥等，回答应该尽量简洁，在三句话以内"}
    ]


def make_messages(messages, input: str, n: int = 20) -> list[dict]:
	"""
	使用 make_messaegs 控制每次请求的消息数量，使其保持在一个合理的范围内，例如默认值是 20。在构建消息列表
	的过程中，我们会先添加 System Prompt，这是因为无论如何对消息进行截断，System Prompt 都是必不可少
	的内容，再获取 messages —— 即历史记录中，最新的 n 条消息作为请求使用的消息，在大部分场景中，这样
	能保证请求的消息所占用的 Tokens 数量不超过模型上下文窗口。
	"""
	# 首先，我们将用户最新的问题构造成一个 message（role=user），并添加到 messages 的尾部
	messages.append({
		"role": "user",
		"content": input,	
	})
 
	# new_messages 是我们下一次请求使用的消息列表，现在让我们来构建它
	new_messages = []
 
	# 每次请求都需要携带 System Messages，因此我们需要先把 system_messages 添加到消息列表中；
	# 注意，即使对消息进行截断，也应该注意保证 System Messages 仍然在 messages 列表中。
	new_messages.extend(system_messages)
 
	# 在这里，当历史消息超过 n 条时，我们仅保留最新的 n 条消息
	if len(messages) > n:
		messages = messages[-n:]
 
	new_messages.extend(messages)
	return new_messages


def chat(client, user_input, history):
    user_content = make_messages(history, user_input)

    completion = client.chat.completions.create(
        model="moonshot-v1-8k",
        messages=user_content,
        temperature=0.3,
    )
    result = completion.choices[0].message
    history.append(result)

    return result.content


# print(chat("月球呢？", history))
if __name__ == "__main__":

    client = OpenAI(
    api_key = "sk-uSS6OrqyCA4XSdUXiPDQyaX5M6wZhwNG6etQx6rUja6IMm47",
    base_url = "https://api.moonshot.cn/v1",
)

    history = []
    
    while True:
        user_input = input()
        print(chat(client, user_input, history))
