from langchain_community.document_loaders import TextLoader # for txt files
from langchain_community.document_loaders import TelegramChatApiLoader # for telegram chat
from langchain_community.document_loaders import PyPDFLoader # for text pdf
from langchain_community.document_loaders import AmazonTextractPDFLoader, UnstructuredImageLoader #for images

from langchain_community.document_loaders import DirectoryLoader


loader1 = TextLoader(
    "reverend_insanity_chapter01.txt",
    encoding="utf-8"
)
docs = loader1.load()
print(docs)
print('------------------')

loader2 = PyPDFLoader('Reverend_insanity2.pdf')
docs2 = loader2.load() 
print(docs2[9])
# [doc,doc..1850 pages] doc = (page_content='bai ning bing', meta_data={'author':'renzu','total_pages':1850})


# loader3 = DirectoryLoader(
#     path='books',
#     glob='*.pdf',
#     loader_cls=PyPDFLoader #which loader to use
#     )

# docs3 = loader3.load()
# print(len(docs3))