import streamlit as st
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate, load_prompt
from dotenv import load_dotenv
load_dotenv()

# Hugging Face model connect API
endpoint = HuggingFaceEndpoint(repo_id='Qwen/Qwen2.5-7B-Instruct')
model = ChatHuggingFace(llm=endpoint)

# Variables for prompt
selected_topic = st.selectbox('select topic',options=['attention','transformers','logistic regression'] )
selected_exp_length = st.selectbox('select exp length',options=['2-3 lines','5 lines',' 8-10 lines'] )
selected_exp_style = st.selectbox('select exp style',options=['simple for begginer','to someone who know ml basics','expert'] )

#prompt template load from json
query_template = load_prompt('query_template.json')

st.set_page_config(page_title='Langchain Prompt')
st.header('Langchain Prompts')

if st.button('get model inference'):
    # Here we are invoking multiple times, instead we can directly use chain to multiple invoke
    # one result is input to next one
    # so prompt_emplate gives prompt after invoke with variables then the prompt is given to model and invoked
    # 
    # prompt = query_template.invoke(
    #     {
    #         'topic':selected_topic,
    #         'exp_style':selected_exp_style,
    #         'exp_length':selected_exp_length
    #     }
    # )
    # result = model.invoke(prompt)
    # st.write(result)

    chain = query_template | model
    result = chain.invoke({
                'topic':selected_topic,
                'exp_style':selected_exp_style,
                'exp_length':selected_exp_length
            })
    st.write(result)





# Result
    # {"content":"### Attention in Data Science\n\n**Explanation Style: Simple for Beginners**\n\nAttention in data science is a technique that helps a model focus on important parts of the input data when making predictions. It’s like a spotlight that highlights the most relevant information.\n\n**Mathematical Details:**\n\nThe attention mechanism uses a weighted sum of the input features. A simple snippet could look like this:\n\n\\[ \\text{Attention}(Q, K, V) = \\text{softmax}\\left(\\frac{QK^T}{\\sqrt{d_k}}\\right)V \\]\n\nHere, \\(Q\\), \\(K\\), and \\(V\\) are matrices representing the query, key, and value respectively. The \\(QK^T\\) term is the dot product between the query and key, which is used to calculate attention scores.\n\n**Analogies:**\n\nThink of attention as a pair of glasses that help you see only the most important things in a crowd. Just like you focus on someone talking to you in a noisy room, the model focuses on the most relevant parts of the data.","additional_kwargs":{},"response_metadata":{"token_usage":{"completion_tokens":220,"prompt_tokens":118,"total_tokens":338},"model_name":"Qwen/Qwen2.5-7B-Instruct","system_fingerprint":null,"finish_reason":"stop","logprobs":null},"type":"ai","name":null,"id":"lc_run--019fd252-affc-7480-bf62-dd3eb0a217c0-0","tool_calls":[],"invalid_tool_calls":[],"usage_metadata":{"input_tokens":118,"output_tokens":220,"total_tokens":338}}