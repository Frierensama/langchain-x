from pydantic import BaseModel, Field, EmailStr
from  typing import Optional, Annotated
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import os
load_dotenv()

llm_endpoint = HuggingFaceEndpoint(
    repo_id=os.getenv('hf_model2')
)
model = ChatHuggingFace(llm=llm_endpoint)

#  create pydantic object to specify schema
class customdict(BaseModel):
    name : str = Field(description='tell the name of the first person in response')
    age : int = Field(description='age of the person')
    gender : Annotated[Optional[str], 'gender of the person']
    email : EmailStr
    cgpa: float = Field(gt=0.0,lt=10.0)

#  create a structured model with schema
structured_model = model.with_structured_output(customdict)

response = structured_model.invoke('Reverend Insanity')
print(response)
