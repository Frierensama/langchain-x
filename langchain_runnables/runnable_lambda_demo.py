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
from langchain_core.runnables import RunnableSequence, RunnableParallel, RunnableBranch, RunnableLambda, RunnablePassthrough


llm_endpoint = HuggingFaceEndpoint(repo_id=os.getenv('hf_model'))
model = ChatHuggingFace(llm=llm_endpoint)
#                     |passthrough-->|
#  give few qoutes--> |              | combine both and result like 'whatever it is 3'-->
#                     |word count-->|
# prompt template
prompt_template = PromptTemplate(
    template='few quotes by fang yuan from Reverend Insanity',
    input_variables=[]
)

def count_words(text):
    return len(text.split())

parser = StrOutputParser()
word_count_runnable = RunnableLambda(count_words)

#  first part
seq_chain = RunnableSequence(prompt_template,model,parser)

parallel_chain = RunnableParallel({
    'passthrough':RunnablePassthrough(),
    'word_counts':word_count_runnable
}
)
# middle part
prompt_template2 = PromptTemplate(
    template='{passthrough} has {word_counts} words',
    input_variables=['passthrough','word_counts']
)
# end part
end_chain = RunnableSequence(prompt_template2,RunnablePassthrough()) #using passthrough, api costly

# merge all
total_chain = RunnableSequence(seq_chain, parallel_chain, end_chain)

result = total_chain.invoke({})
print(result)