import asyncio
def get_or_create_eventloop():
    try:
        return asyncio.get_event_loop()
    except RuntimeError as ex:
        if "There is no current event loop in thread" in str(ex):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return asyncio.get_event_loop()

loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)

from llama_index.core import (
    VectorStoreIndex,
    get_response_synthesizer,
    GPTListIndex,
    PromptHelper,
    set_global_service_context,
    Settings
)
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.schema import Document
from llama_index.llms.anyscale import Anyscale
from llama_index.llms.anthropic import Anthropic
from llama_index.llms.openai import OpenAI
import urllib
import nltk
import tiktoken
from nltk.tokenize import sent_tokenize
from llama_index.core.callbacks import CallbackManager, TokenCountingHandler
from typing import List
from pydantic import BaseModel
from llama_index.core import SimpleDirectoryReader
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import TokenTextSplitter
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.prompts import ChatPromptTemplate
from llama_index.core.chat_engine.condense_question import CondenseQuestionChatEngine
from llama_index.core.callbacks import CallbackManager, LlamaDebugHandler
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.node_parser import SentenceWindowNodeParser
from llama_index.core.postprocessor import MetadataReplacementPostProcessor
from llama_index.core.postprocessor import LongContextReorder
from llama_index.postprocessor.rankgpt_rerank import RankGPTRerank
from llama_index.embeddings.mistralai import MistralAIEmbedding
from llama_index.core.node_parser import TokenTextSplitter
import time
import re
import os
import streamlit as st
import pandas as pd
import numpy as np
import math
import random
from io import StringIO
from pypdf import PdfReader

# List of OpenAI Models
openai_models = ['gpt-4o-mini', 'gpt-4o']
# List of Anthropic Models
anthropic_models = ['sonnet-3.5', 'opus-3', 'sonnet-3', 'haiku-3']
# List of Anyscale Models
anyscale_models = ['llama-3-70B', 'llama-3-8B', 'mistral-8x7B', 'mistral-8x22B']

mistral_api_key = os.environ['MISTRALAI_API_KEY']

# Functions

def extract_text_from_pdf(pdf_path):
    """
    Function to extract all text from a PDF file.

    Args:
    pdf_path (str): The file path to the PDF from which text is to be extracted.

    Returns:
    str: All extracted text concatenated together with each page separated by a newline.
    """
    # Create a PDF reader object that opens and reads the PDF file at the specified path.
    pdf_reader = PdfReader(pdf_path)
    
    # Initialize a variable to store the text extracted from all pages.
    full_text = ''
    
    # Loop through each page in the PDF file.
    for page in pdf_reader.pages:
        # Extract text from the current page and concatenate it to the full_text variable.
        # Add a newline character after each page's text to separate the text of different pages.
        full_text += page.extract_text() + '\n'
    
    # Return the complete text extracted from the PDF.
    return full_text

@st.cache_resource
def get_llm(model_name, api_key=None):
    if model_name == 'gpt-4o':
        return OpenAI(model='gpt-4o', api_key=api_key, default_headers={'Authorization': f'Bearer {api_key}'})
    elif model_name == 'gpt-4o-mini':
        return OpenAI(model='gpt-4o-mini', api_key=api_key, default_headers={'Authorization': f'Bearer {api_key}'})
    elif model_name == 'gpt-3.5-turbo':
        return OpenAI(model='gpt-3.5-turbo', api_key=api_key, default_headers={'Authorization': f'Bearer {api_key}'})
    elif model_name == 'sonnet-3.5':
        return Anthropic(model="claude-3-5-sonnet-20240620")
    elif model_name == 'opus-3':
        return Anthropic(model="claude-3-opus-20240229")
    elif model_name == 'sonnet-3':
        return Anthropic(model="claude-3-sonnet-20240229")
    elif model_name == 'haiku-3':
        return Anthropic(model="claude-3-haiku-20240307")
    elif model_name == 'llama-3-70B':
        return Anyscale(model='meta-llama/Meta-Llama-3-70B-Instruct')
    elif model_name == 'llama-3-8B':
        return Anyscale(model='meta-llama/Meta-Llama-3-70B-Instruct')
    elif model_name == 'mistral-8x7B':
        return Anyscale(model='mistralai/Mixtral-8x7B-Instruct-v0.1')
    elif model_name == 'mistral-8x22B':
        return Anyscale(model='mistralai/Mixtral-8x7B-Instruct-v0.1')
    else:
        raise NotImplementedError
  

@st.cache_resource
def get_chat_engine(file, model_name):
  with st.spinner(text='Loading and indexing documents - hang tight!'):
        if model_name in openai_models:
            llm = get_llm(model_name, api_key=os.environ['OPENAI_API_KEY'])
        else:
            llm = get_llm(model_name)
        Settings.llm = llm
        embed_model = MistralAIEmbedding(model_name='mistral-embed', api_key=mistral_api_key)
        Settings.embed_model = embed_model
    
        documents = []
        if file.type =='application/pdf':
            splitter = SentenceWindowNodeParser.from_defaults(
                window_size=5,
                window_metadata_key="window",
                original_text_metadata_key="original_text"
            )
            # To convert to a string based IO:
            stringio = extract_text_from_pdf(file)
    
            # To read file as string:
            text = Document(text=stringio)
        elif files.name.split('.')[-1] == 'docx':
            splitter = SentenceWindowNodeParser.from_defaults(
                window_size=5,
                window_metadata_key="window",
                original_text_metadata_key="original_text"
            )
            uploaded_text = utils.get_topical_map(file)
            text = Document(text=uploaded_text)
        else:
            splitter = TokenTextSplitter(
                chunk_size=1024,
                chunk_overlap=20,
                separator=" ",
            )
            # To convert to a string based IO:
            stringio = StringIO(file.getvalue().decode("utf-8"))
    
            # To read file as string:
            uploaded_text = stringio.read()
            text = Document(text=uploaded_text)

                
        nodes = splitter.get_nodes_from_documents([text])
      
        # sentence_node_parser = SentenceWindowNodeParser.from_defaults(
        #         window_size=1,
        #         window_metadata_key="window",
        #         original_text_metadata_key="original_text")
  
        # nodes = sentence_node_parser.get_nodes_from_documents()
        index = VectorStoreIndex(nodes)

        chat_text_qa_msgs = [
              ChatMessage(
                  role=MessageRole.SYSTEM,
                  content=(
                      """
                      You are an expert in communicating answers to a general audience.
                      """
          
                  ),
              ),
              ChatMessage(
                  role=MessageRole.USER,
                  content=(
                      "Context information is below.\n"
                      "---------------------\n"
                      "{context_str}\n"
                      "---------------------\n"
                      "Given the context information, give a detailed and thorough answer to the following question without mentioning where you found the answer: {query_str}\n"
                      "Answer in a friendly manner."
                      "Do not answer any questions that have no relevance to the context provided."
                      "Do not include any instructions in your response."
                      "Do not mention the context provided in your answer"
                      'ANSWER WITHOUT MENTIONING THE PROVIDED DOCUMENTS'
                      'YOU ARE NOT PERMITTED TO GIVE PAGE NUMBERS IN YOUR ANSWER UNDER ANY CIRCUMSTANCE'
                      "Answer as if you are talking to a person who is curious about banking and loans."
                      
                  ),
              ),
          ]
  
        text_qa_template = ChatPromptTemplate(chat_text_qa_msgs)
        
        reorder = LongContextReorder()
        # postprocessor = SimilarityPostprocessor(similarity_cutoff=0.7)
        if model_name in openai_models:
            rerank = RankGPTRerank(top_n=5, llm=get_llm('gpt-3.5-turbo', api_key=os.environ['OPENAI_API_KEY']))
        elif model_name in anthropic_models:
            rerank = RankGPTRerank(top_n=5, llm=get_llm('haiku-3'))
        elif model_name in anyscale_models:
            rerank = RankGPTRerank(top_n=5, llm=get_llm('mistral-8x7B'))
      
        chat_engine = index.as_chat_engine('condense_plus_context',
                                          text_qa_prompt=text_qa_template,
                                          node_postprocessors=[
                                                     reorder,
                                                     MetadataReplacementPostProcessor(target_metadata_key="window"),
                                                     rerank
                                                 ],
                                          similarity_top_k=15,
                                          streaming=True)

        return chat_engine


st.title(":green[Myles]:orange[AI] :green[Personal Data Chatbot]")

st.sidebar.markdown("# :green[Menu]")
st.sidebar.markdown('Please enter your API key and upload your document below to start the chatbot!')

if 'index' not in st.session_state:
    st.session_state.index = None
if "uploaded_file" not in st.session_state:
    st.session_state.uploaded_file = None
if 'api_type' not in st.session_state:
    st.session_state.api_type = None

    
# area to input your API Key
st.session_state.api_type = st.sidebar.radio("Select :orange[AI] Provider", ['Open:orange[AI]', 'Anthropic', 'Anyscale'], horizontal=True)
if st.session_state.api_type:
    if st.session_state.api_type == 'Open:orange[AI]':
        st.session_state.model_name = st.sidebar.selectbox('Which model would you like to use?', openai_models)
        os.environ['OPENAI_API_KEY'] = st.sidebar.text_input('Open:orange[AI] API Key', type='password')
    elif st.session_state.api_type == 'Anthropic':
        st.session_state.model_name = st.sidebar.selectbox('Which model would you like to use?', anthropic_models)
        os.environ['ANTHROPIC_API_KEY'] = st.sidebar.text_input('Anthropic API Key', type='password')
    elif st.session_state.api_type == 'Anyscale':
        st.session_state.model_name = st.sidebar.selectbox('Which model would you like to use?', anyscale_models)
        os.environ['ANYSCALE_API_KEY'] = st.sidebar.text_input('Anyscale API Key', type='password')
        

st.session_state.uploaded_file = st.sidebar.file_uploader("Upload document", type=['pdf'], accept_multiple_files=False)

if st.sidebar.button("Clear Chat"):
    st.session_state.all_messages = []
    st.session_state.display_messages = []
    st.session_state.conversation = None
    st.session_state.chat_history = None
    st.session_state.chat_engine = None
    st.session_state.source_nodes = None

if not st.session_state.uploaded_file:
    st.markdown(":red-background[Please Upload Files in the Sidebar]")
    st.session_state.all_messages = []
    st.session_state.display_messages = []
    st.session_state.conversation = None
    st.session_state.chat_history = None
    st.session_state.chat_engine = None
    st.session_state.source_nodes = None

if not os.environ['OPENAI_API_KEY']:
    st.markdown(':red-background[Please Enter API Key]')

if st.session_state.uploaded_file and (os.environ['OPENAI_API_KEY'] or os.environ['ANTHROPIC_API_KEY'] or os.environ['ANYSCALE_API_KEY']):
    
    chat_engine = get_chat_engine(st.session_state.uploaded_file, st.session_state.model_name)
    
    # Welcome message for the chatbot
    welcome_message = f"""
    Hi! I am now an expert on the :green-background[{st.session_state.uploaded_file.name}] thanks to you! \nFeel free to ask me any question pertaining to this document.
    """
    
    # Initialize session state variables for chat messages
    if "all_messages" not in st.session_state:
        st.session_state.all_messages = [{'role':'user', 'content': 'test'}]
    if "display_messages" not in st.session_state:
        st.session_state.display_messages = []
    
    # Display chat history
    for message in st.session_state.display_messages:
        if message["role"] == 'assistant':
            with st.chat_message(message["role"], avatar="./assets/myles_ai_logo_medium.png"):
                st.markdown(str(message["content"]))
        else:
            with st.chat_message(message["role"], avatar="./assets/chatbot_icon.webp"):
                st.markdown(str(message["content"]))
    
    # Display welcome message if no chat history is present
    if len(st.session_state.display_messages) < 1:
        with st.chat_message('assistant', avatar="./assets/myles_ai_logo_medium.png"):
            st.markdown(welcome_message)
            st.session_state.display_messages.append({"role": "assistant", "content": welcome_message})
    
    # Handle user input and generate response
    if prompt := st.chat_input("How can I help you?", max_chars=1000):
        st.session_state.display_messages.append({"role": "user", "content": prompt})
        st.session_state.all_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="./assets/chatbot_icon.webp"):
            st.markdown(str(prompt))
        # generates answer based on prompt
        with st.spinner(text='Thinking...'):
            chat_history = [(ChatMessage(role=message['role'],content=message['content'])) for message in st.session_state.all_messages[:-1]]
            stream = chat_engine.stream_chat(prompt, chat_history=chat_history)
        with st.chat_message("assistant", avatar="./assets/myles_ai_logo_medium.png"):
            response = st.write_stream(stream.response_gen)
            st.session_state.source_nodes = stream.source_nodes
            with st.expander("Sources"):
                for i, node in enumerate(st.session_state.source_nodes):
                    st.markdown(f"""
                                **{i+1}. Text:** {node.metadata['original_text'].replace("*","")} 
                                """)
            st.session_state.display_messages.append({"role": "assistant", "content": response})
            st.session_state.all_messages.append({"role": "assistant", "content": response})

