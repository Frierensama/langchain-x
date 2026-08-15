from dotenv import load_dotenv
import os

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


load_dotenv()

llm = HuggingFaceEndpoint(repo_id=os.getenv('hf_model'))
hf_model = ChatHuggingFace(llm=llm)

prompt_template1 = PromptTemplate(
    template='you are an AI assistant about magic. explain about the {topic} in detailed report',
    input = ['topic']
)
prompt_template2 = PromptTemplate(
    template='five line summary about {text}',
    input = ['text']
)

parser = StrOutputParser()
chain = prompt_template1 | hf_model | prompt_template2 | hf_model | parser

chain_result = chain.invoke({'topic' : "chronostatis"})
print(chain_result)


# 1. **Definition**: Chronostatis is a magical term for a state where an individual or object remains suspended in time, unaffected by the passage of time.
# 2. **Properties**: It involves time inversion, temporal immunity, and acts as a spectator mode for observing events.
# 3. **Influence**: Observing in this state can subtly alter events, though this is more metaphysical.
# 4. **Applications**: Used in historical investigation, espionage, and time traveling.
# 5. **Concept**: A spell or condition that allows for passive observation and potential influence on temporal events.

