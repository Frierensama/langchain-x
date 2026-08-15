from typing import TypedDict, Annotated, Optional
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
import os
load_dotenv()

llm_endpoint = HuggingFaceEndpoint(
    repo_id=os.getenv('hf_model')
)
model = ChatHuggingFace(llm=llm_endpoint)

# create pydantic object to specif schema
class customdict(TypedDict):
    name: Annotated[ str,"the name of the reviewer" ]
    age: Annotated[Optional[int], 'the age of the reviewer']

# create structured output model 
structured_model  = model.with_structured_output(customdict)
result = structured_model.invoke('Reverend Insanity')

#  result - dict , model should support the pydantic
print(result)
