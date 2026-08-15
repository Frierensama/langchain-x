from pydantic import BaseModel, Field
from typing import Optional, Annotated
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate

from dotenv import load_dotenv
import os
load_dotenv()

llm_endpoint = HuggingFaceEndpoint(
    repo_id=os.getenv('hf_model')
)
hf_model = ChatHuggingFace(llm=llm_endpoint)

#  create pydantic object to specify the schema
class customschema(BaseModel):
    name : str
    age : int = Field(description='age of the character')
    morality_quotes: list[str] = Field(description='top 10 quotes by fang yuan')

parser = PydanticOutputParser(pydantic_object=customschema)

# for non-structured output, attach the schema parser in template
prompt_template = PromptTemplate(
    template='tell me about the anime {anime_name} {format_instructions}' ,
    input_variables=['anime_name'],
    partial_variables={'format_instructions':parser.get_format_instructions()}
)

chain = prompt_template | hf_model | parser

result = chain.invoke({'anime_name':'Reverend Insanity'})
print(result)