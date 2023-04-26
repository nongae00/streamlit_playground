import os 
import sys

import streamlit as st 
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain 
from langchain.memory import ConversationBufferMemory
from langchain.utilities import WikipediaAPIWrapper 

from langchain import HuggingFaceHub
#from langchain.memory import ConversationBufferWindowMemory

os.environ['OPENAI_API_KEY'] = st.secrets["apikey"]
os.environ["HUGGINGFACEHUB_API_TOKEN"] =  st.secrets["hf_token"]


from langchain import PromptTemplate, HuggingFaceHub, LLMChain
template = """Question: {question}

Answer: Let's think step by step."""
question_prompt = PromptTemplate(template=template, input_variables=["question"])
llm=HuggingFaceHub(repo_id="google/flan-t5-xl", model_kwargs={"temperature":1e-10})

question = "When was Google founded?"

#title_memory = ConversationBufferMemory(input_key='topic', memory_key='chat_history')
question_memory = ConversationBufferMemory(input_key='title', memory_key='chat_history')


# Llms
#script_chain = LLMChain(llm=llm, prompt=script_template, verbose=True, output_key='script', memory=script_memory)

question_chain = LLMChain(prompt=question_prompt, llm=llm, verbose=True, memory=question_memory)

st.title('🦜🔗 Top 10 Resources')

print(question_chain.run(question))
st.write(question_chain.run(question))


'''
hf_llm = HuggingFaceHub(repo_id="google/flan-t5-xl", model_kwargs={"temperature":0, "max_length":512})
k = 10

# App framework
st.title('🦜🔗 Top 10 Resources')
prompt = st.text_input('Plug in your prompt here') 

# Prompt templates
title_template = PromptTemplate(
    input_variables = ['topic'], 
    template='write me a youtube video title about {topic}'
)

script_template = PromptTemplate(
    input_variables = ['title', 'wikipedia_research'], 
    #template='write me a youtube video script based on this title TITLE: {title} while leveraging this wikipedia reserch:{wikipedia_research} '
    template='What are the top 10 resources to learn {title} in 2023 while leveraging this wikipedia research:{wikipedia_research}?'
    
)

hf_prompt = PromptTemplate(
    input_variables=["topic"],
    template="translate English to Korean: {topic}"
)


# Memory 
title_memory = ConversationBufferMemory(input_key='topic', memory_key='chat_history')
script_memory = ConversationBufferMemory(input_key='title', memory_key='chat_history')


# Llms
llm = OpenAI(temperature=0.9) 
title_chain = LLMChain(llm=llm, prompt=title_template, verbose=True, output_key='title', memory=title_memory)
script_chain = LLMChain(llm=llm, prompt=script_template, verbose=True, output_key='script', memory=script_memory)
hf_chain = LLMChain(llm=hf_llm, prompt=hf_prompt, verbose=True, output_key='translate', memory=script_memory) 
#memory=ConversationBufferWindowMemory(k=2))

wiki = WikipediaAPIWrapper()

# Show stuff to the screen if there's a prompt
if prompt: 
    translate = hf_chain.run(prompt)
    st.write(translate)
    
    #sys.exit()
'''

'''title = title_chain.run(prompt)
    wiki_research = wiki.run(prompt) 
    script = script_chain.run(title=title, wikipedia_research=wiki_research)
    st.write(title) 
    #st.write(hf_llm("translate English to Korean: " + title))
    st.write(script) 
    #st.write(hf_llm("translate English to Korean: " + script))
    with st.expander('Title History'): 
        st.info(title_memory.buffer)
    with st.expander('Script History'): 
        st.info(script_memory.buffer)
    with st.expander('Wikipedia Research'): 
        st.info(wiki_research)
'''
