from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
load_dotenv()

llm = HuggingFaceEndpoint(repo_id='meta-llama/Llama-3.1-8B-Instruct')
hf_model = ChatHuggingFace(llm= llm)

result = hf_model.invoke('when does saturn give reward of sadesati phase')
print(result)

# content='The capital of Japan is Tokyo.' additional_kwargs={} 
# response_metadata={'token_usage': {'completion_tokens': 8, 'prompt_tokens': 16, 'total_tokens': 24},
#                    'model_name': 'meta-llama/Llama-3.1-8B-Instruct',
#                    'system_fingerprint': None, 'finish_reason': 'stop', 'logprobs': None
                    # }
# id='lc_run--019fd0dc-1247-7ee3-a519-58fe47ef1e28-0' 
# tool_calls=[] 
# invalid_tool_calls=[] 
# usage_metadata={'input_tokens': 16, 'output_tokens': 8, 'total_tokens': 24}