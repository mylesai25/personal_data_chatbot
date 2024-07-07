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
from llama_index.core.indices.service_context import ServiceContext
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

for k, v in st.session_state.items():
    st.session_state[k] = v

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
def get_api_type(api_type):
    if api_type == 'openai':
        # default is gpt-3.5-turbo, can also be gpt-4-0314
        return OpenAI(model='gpt-4o') # for QA, temp is low
    elif api_type == 'claude':
        return Anthropic(model="claude-3-opus-20240229")
    elif api_type == 'llama':
        return Anyscale(model='meta-llama/Llama-2-70b-chat-hf')
    elif api_type == 'mistral':
        return Anyscale(model='mistralai/Mixtral-8x7B-Instruct-v0.1', max_tokens=10000)
    else:
        raise NotImplementedError
  

@st.cache_resource
def get_chat_engine(files):
  with st.spinner(text='Loading and indexing documents - hang tight!'):
        
        llm = get_api_type('openai')
        Settings.llm = llm
        embed_model = MistralAIEmbedding(model_name='mistral-embed', api_key=mistral_api_key)
        Settings.embed_model = embed_model
        

        splitter = TokenTextSplitter(
            chunk_size=1024,
            chunk_overlap=20,
            separator=" ",
        )
    
        documents = []
        for uploaded_file in files:
            if uploaded_file.type =='application/pdf':
                # To convert to a string based IO:
                stringio = extract_text_from_pdf(uploaded_file)
        
                # To read file as string:
                text = Document(text=stringio)
            elif uploaded_file.name.split('.')[-1] == 'docx':
                uploaded_text = utils.get_topical_map(uploaded_file)
                text = Document(text=uploaded_text)
            else:
                # To convert to a string based IO:
                stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
        
                # To read file as string:
                uploaded_text = stringio.read()
                text = Document(text=uploaded_text)
            documents.append(text)

                
        nodes = splitter.get_nodes_from_documents(documents)
      
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
                      You are an expert on developing websites for contractors and explaining your expertise to a general audience.
                      If a character or word limit is mentioned in the prompt, ADHERE TO IT. 
                      For example, if a user wants a summary of a business less than 750 characters, the summary must be less than 750 characters.
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
        rerank = RankGPTRerank(top_n=5, llm=Anthropic(model="claude-3-haiku-20240307"))
      
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


st.sidebar.markdown("# Menu")
st.sidebar.markdown('Please enter your API KEY and upload your document below to start the chatbot!')

if st.sidebar.button("Clear Chat"):
    st.session_state.messages = []
    st.session_state.conversation = None
    st.session_state.chat_history = None
    st.session_state.file_name =  None
    st.session_state.page = None
    st.session_state.chat_engine = None


if 'index' not in st.session_state:
    st.session_state.index = None
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []

    
# area to input your API Key
os.environ['OPENAI_API_KEY'] = st.sidebar.text_input('OpenAI API Key', type='password')

st.session_state.uploaded_files = st.sidebar.file_uploader("Upload document", type=['docx', 'pdf'], accept_multiple_files=False)

if not st.session_state.uploaded_files:
    st.markdown("Please Upload Files in the Sidebar")

if not os.environ['OPENAI_API_KEY']:
    st.markdown('Please Enter API Key')

if st.session_state.uploaded_files and os.enviorn['OPENAI_API_KEY']:

    st.sidebar.markdown(f'Uploaded Documents: \n{[file.name for file in st.session_state.uploaded_files]}')
    
    chat_engine = get_chat_engine(st.session_state.uploaded_files)
    
    # initializes messages for chatbot
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    
    # displays chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
        
    
    
    # start of prompt
    if prompt := st.chat_input("How can I help you?"):
        st.session_state.messages.append({"role": "user", "content": prompt})  
        with st.chat_message("user"):
            st.markdown(prompt)
        # generates answer based on prompt
        with st.spinner(text='Thinking...'):
            chat_history = [(ChatMessage(role=message['role'],content=message['content'])) for message in st.session_state.messages[:-1]]
            stream = chat_engine.stream_chat(prompt, chat_history=chat_history)
        with st.chat_message("assistant"):
        
            response = st.write_stream(stream.response_gen)
        st.session_state.messages.append({'role': 'assistant', 'content': response})

