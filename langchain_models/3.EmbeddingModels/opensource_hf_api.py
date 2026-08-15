from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpointEmbeddings
from dotenv import load_dotenv
load_dotenv()

embeddings = HuggingFaceEndpointEmbeddings(
    model='Qwen/Qwen3-Embedding-8B'
)

query_result = embeddings.embed_query("This is a test document.")
print(query_result)