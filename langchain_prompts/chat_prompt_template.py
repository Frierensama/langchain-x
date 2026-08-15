from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
load_dotenv()

llm_endpoint = HuggingFaceEndpoint(repo_id='Qwen/Qwen2.5-7B-Instruct')
qwen_model = ChatHuggingFace(llm=llm_endpoint)

chat_template = ChatPromptTemplate([
    ('system','you are an ai assistant'),
    MessagesPlaceholder(variable_name='chat_history'),
    ('human','{prompt}')
])

chat_history = []

with open('chat_history.txt','r') as file:
    chat_history.extend(file.readlines())

after_days_prompt = chat_template.invoke({'chat_history':chat_history,'prompt':'where is my refund. its been 6 days.'})

output = qwen_model.invoke(after_days_prompt)
print(output)

# content="I understand your concern. It usually takes 3-5 days to process a refund once we receive your request. Since it's been 6 days, it might be a good idea to check the payment method you used for the refund, as the processing time can vary depending on your bank or payment processor. Alternatively, you can reach out to our customer support team for a faster resolution. They can provide you with more specific details about the status of your refund. Would you like me to help you contact customer support?" additional_kwargs={} response_metadata={'token_usage': {'completion_tokens': 105, 'prompt_tokens': 65, 'total_tokens': 170}, 'model_name': 'Qwen/Qwen2.5-7B-Instruct', 'system_fingerprint': None, 'finish_reason': 'stop', 'logprobs': None} id='lc_run--019fd62b-c55c-71b3-926d-543947282702-0' tool_calls=[] invalid_tool_calls=[] usage_metadata={'input_tokens': 65, 'output_tokens': 105, 'total_tokens': 170}