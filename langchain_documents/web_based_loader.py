from langchain_community.document_loaders import WebBaseLoader

url = 'https://huggingface.co/Qwen/Qwen2.5-7B-Instruct'
loader = WebBaseLoader(url)
print(loader.load())
