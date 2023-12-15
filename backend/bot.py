# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import asyncio
import os
import random
import re
import requests
import json
import openai

from botbuilder.core import ActivityHandler, TurnContext
from botbuilder.schema import Activity, ActivityTypes, ChannelAccount
from langchain.chains import LLMChain
from langchain.chat_models import AzureChatOpenAI

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders import TextLoader


from botbuilder.schema import (
    ConversationAccount,
    Attachment,
)
from botbuilder.schema.teams import (
    FileDownloadInfo,
    FileConsentCard,
    FileConsentCardResponse,
    FileInfoCard,
)
from botbuilder.schema.teams.additional_properties import ContentType


from langchain.memory import (ConversationBufferMemory,
                              ConversationBufferWindowMemory,
                              CosmosDBChatMessageHistory)
from langchain.prompts import (ChatPromptTemplate, HumanMessagePromptTemplate,
                               MessagesPlaceholder,
                               SystemMessagePromptTemplate)

from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.docstore.document import Document

from prompts import (CUSTOM_CHATBOT_PREFIX, WELCOME_MESSAGE)
from prompts import COMBINE_QUESTION_PROMPT, COMBINE_PROMPT


# from utils import get_search_results

from typing import Any, Dict, List, Optional, Awaitable, Callable, Tuple, Type, Union
from collections import OrderedDict
import uuid

import markdownify


import requests
from typing import Dict, List

from langchain.chat_models import AzureChatOpenAI
from langchain.agents import AgentExecutor, Tool, ZeroShotAgent
from langchain.callbacks.manager import CallbackManager
from langchain.agents import initialize_agent, AgentType
from langchain.tools import BaseTool
from langchain.utilities import BingSearchAPIWrapper

from callbacks import StdOutCallbackHandler
from prompts import BING_PROMPT_PREFIX

# from IPython.display import Markdown, HTML, display  

# from dotenv import load_dotenv
# load_dotenv("credentials.env")


# GPT-4 models are necessary for this feature. GPT-35-turbo will make mistakes multiple times on following system prompt instructions.
MODEL_DEPLOYMENT_NAME = "gpt-4-turbo" 
cb_handler = StdOutCallbackHandler()
cb_manager = CallbackManager(handlers=[cb_handler])
#####################


# Env variables needed by langchain
# os.environ["OPENAI_API_BASE"] = os.environ.get("AZURE_OPENAI_ENDPOINT")
# os.environ["OPENAI_API_KEY"] = os.environ.get("AZURE_OPENAI_API_KEY")
os.environ["OPENAI_API_VERSION"] = os.environ.get("AZURE_OPENAI_API_VERSION")
# os.environ["OPENAI_API_TYPE"] = "azure"

class MyBingSearch(BaseTool):
    """Tool for a Bing Search Wrapper"""
    
    name = "@bing"
    description = "useful when the questions includes the term: @bing.\n"

    k: int = 5
    
    def _run(self, query: str) -> str:
        bing = BingSearchAPIWrapper(k=self.k)
        return bing.results(query,num_results=self.k)
            
    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("This Tool does not support async")
      
# Bot Class
class MyBot(ActivityHandler):
    memory = None
    prompt = ChatPromptTemplate(
                messages=[
                    SystemMessagePromptTemplate.from_template(
                        CUSTOM_CHATBOT_PREFIX
                    ),
                    # The `variable_name` here is what must align with memory
                    MessagesPlaceholder(variable_name="chat_history"),
                    HumanMessagePromptTemplate.from_template("{question}")
                ]
            )
    
    memory_dict = {}
    memory_cleared_messages = [
        "Historie byla smazána.",
        "Historie byla vynulována.",
        "Historie byla obnovena na výchozí hodnoty.",
        "Došlo k resetování historie.",
        "Historie byla resetována na počáteční stav."
    ]



    def __init__(self):
        self.model_name = os.environ.get("AZURE_OPENAI_MODEL_NAME") 
        self.llm = AzureChatOpenAI(deployment_name=self.model_name, temperature=0, max_tokens=1000)  
   
   
    # format the response (convert html to markdown)
    def format_response(self, response):
        # return re.sub(r"(\n\s*)+\n+", "\n\n", response).strip()

        # convert html tags to markdown
        response = markdownify.markdownify(response, heading_style="ATX")
    
        return response.strip()
    
    # ask GPT    
    def ask_gpt(self, session_id):
        # Now we declare our LLM object with the callback handler 

        tools = [MyBingSearch(k=5)]
        agent_executor = initialize_agent(tools, self.llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, 
                                agent_kwargs={'prefix':BING_PROMPT_PREFIX}, callback_manager=cb_manager)

        # suffix = """
        #     Begin!"

        #     {chat_history}
        #     Question: {input}
        #     {agent_scratchpad}"""
        # prompt = ZeroShotAgent.create_prompt(
        #     tools,
        #     prefix=BING_PROMPT_PREFIX,
        #     suffix=suffix,
        #     input_variables=["input", "chat_history", "agent_scratchpad"],
        # )
        # llm_chain = LLMChain(llm=self.llm, prompt=prompt)
        # agent = ZeroShotAgent(llm_chain=llm_chain, tools=tools, verbose=True)
        # agent_chain = AgentExecutor.from_agent_and_tools(
        #     agent=agent, tools=tools, verbose=True, memory=self.memory_dict[session_id]
        # )

        QUESTION = self.QUESTION.strip()

        #As LLMs responses are never the same, we do a for loop in case the answer cannot be parsed according to our prompt instructions
        for i in range(2):
            try:
                response = agent_executor.run(QUESTION) 
                # response = agent_chain.run(QUESTION)
                break
            except Exception as e:
                response = str(e)
                continue
        return self.format_response(response)
    
   

    # Function to show welcome message to new users
    async def on_members_added_activity(self, members_added: ChannelAccount, turn_context: TurnContext):
        for member_added in members_added:
            if member_added.id != turn_context.activity.recipient.id:
                await turn_context.send_activity(WELCOME_MESSAGE + "\n\n" + self.model_name)
    
    
    # See https://aka.ms/about-bot-activity-message to learn more about the message and other activity types.
    async def on_message_activity(self, turn_context: TurnContext):
             
        # Extract info from TurnContext - You can change this to whatever , this is just one option 
        session_id = turn_context.activity.conversation.id
        user_id = turn_context.activity.from_property.id + "-" + turn_context.activity.channel_id

        self.QUESTION = turn_context.activity.text

        if session_id not in self.memory_dict:
            self.memory_dict[session_id] = ConversationBufferWindowMemory(memory_key="chat_history",input_key="question", return_messages=True, k=3)

        if (self.QUESTION.strip() == "/reset"):
            # self.memory.clear()
            self.memory_dict[session_id].clear()
            self.db = None # reset the db
            # randomly pick one of the memory_cleared_messages
            await turn_context.send_activity(random.choice(self.memory_cleared_messages))
            # await turn_context.send_activity("Memory cleared")
        elif (self.QUESTION.strip() == "/help"):
            await turn_context.send_activity(WELCOME_MESSAGE + "\n\n" + self.model_name)
        else:    
            answer = self.ask_gpt(session_id)
            await turn_context.send_activity(answer)





