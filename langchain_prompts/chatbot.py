from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from dotenv import load_dotenv
load_dotenv()

llm_endpoint = HuggingFaceEndpoint(repo_id='Qwen/Qwen2.5-7B-Instruct')
qwen_model = ChatHuggingFace(llm=llm_endpoint)

messages = [
    SystemMessage(content='you are an ai assistant')
]

while True:
    user_input = input('you:')
    if user_input == 'exit':
        break

    messages.append(HumanMessage(content=user_input))
    output = qwen_model.invoke(messages)
    print('AI:'+ output.content)
    messages.append(AIMessage(content=output.content))

print(messages)