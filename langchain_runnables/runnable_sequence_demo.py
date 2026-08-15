from dotenv import load_dotenv
import os
load_dotenv()
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint #  llm api
# schema output
from pydantic import BaseModel, Field
from typing import TypedDict, Optional, Annotated
# parsers, prompt
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.prompts import PromptTemplate
# runnable types, few
from langchain_core.runnables import RunnableSequence, RunnableParallel, RunnableBranch


llm_endpoint = HuggingFaceEndpoint(repo_id=os.getenv('hf_model'))
model = ChatHuggingFace(llm=llm_endpoint)

# prompt templates
prompt_template1 = PromptTemplate(
    template='Give me 3 main character names from {anime_name}',
    input_variables=['anime_name']
)
prompt_template2 = PromptTemplate(
    template='Give me one quote for each {character_name}',
    input_variables=['character_name']
)

parser = StrOutputParser()

seq_chain = RunnableSequence(prompt_template1, model, parser, prompt_template2, model, parser)

result = seq_chain.invoke({'anime_name':'naruto'})
print(result)

# It seems like you're asking for quotes from each of the main characters in the Naruto series. Here are some well-known quotes for each of them:

# 1. **Naruto Uzumaki**
#    - "Everyone has their own way of becoming strong. I became a ninja so that no one would have to look down on me and my village ever again. That’s my dream. A dream that will come true."

# 2. **Sasuke Uchiha**
#    - "I won't be the one left behind. I won't be the last! I'll become stronger and when I meet you again, you'll be the one trailing behind me!"

# 3. **Sakura Haruno**
#    - "I can't let you go. I can't allow you to just leave me behind. No matter where you go, I'll be there. I'm not going to lose you, Sasuke."

# These quotes capture some of the key themes and motivations of each character in the Naruto series.