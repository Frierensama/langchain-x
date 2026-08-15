from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()

openai_chatmodel = ChatOpenAI(model='gpt-3.5-turbo-instruct')

result = openai_chatmodel.invoke('hello')
print(result)