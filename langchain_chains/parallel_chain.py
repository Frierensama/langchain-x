from langchain_core.prompts import PromptTemplate
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.runnables import RunnableParallel

from pydantic import BaseModel, Field
from typing import TypedDict, Annotated, Optional

from dotenv import load_dotenv
import os
load_dotenv()

# just using two different models , no reason
llm_endpoint1 = HuggingFaceEndpoint(repo_id=os.getenv('hf_model'))
hf_model1 = ChatHuggingFace(llm=llm_endpoint1)

llm_endpoint2 = HuggingFaceEndpoint(repo_id=os.getenv('hf_model2'))
hf_model2 = ChatHuggingFace(llm=llm_endpoint2)

# three prompts 1.description 2.quotes 3.merge both
prompt_template1 = PromptTemplate(
    template='give me a detailed description about the anime {anime_name}',
    input_variables=['anime_name']
)
prompt_template2 = PromptTemplate(
    template='top 10 quotes from {char_name}',
    input_variables=['char_name']
)
prompt_template3 = PromptTemplate(
    template = 'merge the both anime details {description} and quotes {quotes}',
    input_variables=['description','quotes']
)

parser = StrOutputParser()

# parallel chain returns dict with key values
parallel_chain = RunnableParallel(
    {
    'description' : prompt_template1 | hf_model1 | parser,
    'quotes' : prompt_template2 | hf_model2 | parser
    }
)
# second part
merge_chain = prompt_template3 | hf_model1 | parser
# top and bottom connect
total_chain = parallel_chain | merge_chain

result = total_chain.invoke(
    {
        'anime_name': "Reverend Insanity",
        'char_name' : "Fang Yuan"
    }
)

print(result)

# schema for output 
# class customdict(TypedDict):
#     name:Annotated[str,'name of the main character']
#     quotes:Annotated[list[str], 'top quotes by main character']
