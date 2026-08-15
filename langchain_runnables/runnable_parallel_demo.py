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
    template='give 5 popular anime names',
    input_variables=[]
)
prompt_template2 = PromptTemplate(
    template='Give 5 popular anime character names',
    input_variables=[]
)

parser = StrOutputParser()

seq_chain = RunnableParallel(
    {
        'anime_names': RunnableSequence(prompt_template1,model,parser),
        'character_names':RunnableSequence(prompt_template2,model,parser)
    }
)
prompt_template3 = PromptTemplate(
    template='check if anime of the characters -{character_names} are from any one of these animes {anime_names}. print the character_names which i provided and anime_names too at the starting before explaination',
    input_variables=['character_names','anime_names']
)
merg_chain = RunnableSequence(seq_chain,prompt_template3, model, parser)

result = merg_chain.invoke({})
print(result)


# 当然可以！以下是您提供的角色和对应的动漫作品，以及对这些作品特点的更具体描述：

# ### 动漫角色和作品

# 1. **Sasuke Uchiha** - 《火影忍者》（Naruto）
# 2. **Kazuki Takahashi** - 《伪装者 隐者》（Kagemusha the Hidden Samurai）
# 3. **Yato** - 《灵人灵马神》（Noragami）
# 4. **Misato Katsuragi** - 《新世纪福音战士》（Neon Genesis Evangelion）
# 5. **Ichigo Kurosaki** - 《BLEACH》（ bleach）

# ### 作品特点描述

# 1. **《火影忍者》（Naruto）**
#    - **类型**: 动作、冒险、科幻
#    - **特点**: 以日本忍者世界为背景，讲述主角漩涡鸣人在成长和守护同伴、结交朋友的过程中逐渐成为强大忍者的故事。作品中包含丰富的角色发展和激烈的忍术对决。
   
# 2. **《伪装者 隐者》（Kagemusha the Hidden Samurai）**
#    - **类型**: 武侠、历史
#    - **特点**: 以日本战国时代为背景，讲述了主人公如何通过自己的智慧和勇气在复杂的政局中生存的故事。作品中融合了武士道精神和人性的探讨。
   
# 3. **《灵人灵马神》（Noragami）**
#    - **类型**: 神话、都市奇幻
#    - **特点**: 以现代都市为背景，讲述一位被称为“灵马神”的神明如何在现实世界中解决烦恼、帮助他人，同时也探索了神明与人类之间的关系。
   
# 4. **《新世纪福音战士》（Neon Genesis Evangelion）**
#    - **类型**: 科幻、心理
#    - **特点**: 以未来世界为背景，围绕一群高中生与巨大生物“使徒”战斗的故事。作品中充满了深刻的心理描写和哲学思考，探讨了人类存在的意义和生命的本质。
   
# 5. **《BLEACH》（ bleach）**
#    - **类型**: 动作、冒险、超自然
#    - **特点**: 以日本为背景，讲述高中生速 Master (Soul Reaper) 保护人类免受超自然生物威胁的故事。作品中结合了丰富的战斗场面和深刻的人物