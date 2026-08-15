from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv
load_dotenv()

anthropic_model = ChatAnthropic(model='whatever-model-try')

result = anthropic_model.invoke('who are you')
print(result)