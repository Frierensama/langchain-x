from langchain_core.prompts import PromptTemplate

query_template = PromptTemplate(
    template="""
    prepare a summary for {topic} in data science with following specifications:
    Explaination style : {exp_style}
    Explaination length: {exp_length}
    1.Mathematical Details:
    include relevant mathematical equations, simple snippets
    2.analogies:
    simplify complex analogies

    if you dont know about the topic in detailed, tell i dont have enough data to provide the infromation.
""",
input_variables=['topic','exp_style','exp_length']
)

query_template.save('query_template.json')