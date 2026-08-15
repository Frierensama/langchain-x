from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader

# text_loader = TextLoader('reverend_insanity_chapter01.txt',encoding='utf-8')
# docs = text_loader.load()

# # splitter = CharacterTextSplitter(chunk_size=500,chunk_overlap=20,separator='')
# # splits = splitter.split_text(text)
# # print(splits)

pdf_loader = PyPDFLoader('Reverend_Insanity2.pdf')
docs = pdf_loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=0)
splits = splitter.split_documents(docs)

print(splits[11])
print('_________________')
print(splits[12])
