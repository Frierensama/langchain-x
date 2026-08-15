from dotenv import load_dotenv
import os
load_dotenv()

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

llm = HuggingFaceEndpoint(repo_id=os.getenv('hf_model'))
hf_model = ChatHuggingFace(llm=llm)

parser = JsonOutputParser()

prompt_template = PromptTemplate(
    template='give me name, age, title, alias of the main character of the anime {anime_name} as {format_instructions}',
    input_variables = ['anime_name'],
    partial_variables={'format_instructions' : parser.get_format_instructions()}
)

chain = prompt_template | hf_model | parser

result = chain.invoke({'anime_name':'Redo of healer'})
print(result)
