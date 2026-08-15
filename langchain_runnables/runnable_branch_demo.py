from langchain_core.prompts import PromptTemplate
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableBranch

from pydantic import BaseModel, Field
from typing import TypedDict, Annotated, Optional, Literal

from dotenv import load_dotenv
import os
load_dotenv()

llm_endpoint1 = HuggingFaceEndpoint(repo_id=os.getenv('hf_model'))
hf_model1 = ChatHuggingFace(llm=llm_endpoint1)

llm_endpoint2 = HuggingFaceEndpoint(repo_id=os.getenv('hf_model2'))
hf_model2 = ChatHuggingFace(llm=llm_endpoint2)

# three prompts 1.description 2.quotes 3.merge both
prompt_template1 = PromptTemplate(
    template='give me 10 names from the anime {anime_name}',
    input_variables=['anime_name']
)
prompt_template2 = PromptTemplate(
    template='10 alias names for {char_name}',
    input_variables=['char_name']
)

parse = StrOutputParser()

class custom_schema(TypedDict):
    sentiment: Literal['positive','negative']
    char_name: str
    anime_name: str

classifier_chain = hf_model1.with_structured_output(schema= custom_schema)

branch_chain = RunnableBranch(
    (lambda x : x['sentiment'] == 'positive', prompt_template1 | hf_model1 | parse ),
    (lambda x : x['sentiment'] == 'negative', prompt_template2 | hf_model2 | parse ),
    prompt_template1 | hf_model1 | parse
)

chain = classifier_chain | branch_chain
result = chain.invoke('pick either positive or negative, then character name is fang yuan, anime_name is Pokemon')

print(result)