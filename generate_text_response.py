import streamlit as st
from langchain_core.messages import HumanMessage,AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import tool
from langgraph.prebuilt import create_react_agent
import base64
import io
from PIL import Image
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
import os
load_dotenv()
def generate_text(user_input):
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=os.getenv("GEMINI_API_KEY"),
        temperature=0.5
    )
    tools = []
    agent_execute = create_react_agent(model=llm,tools=tools)


    response_result = ''
    for chunk in agent_execute.stream({"messages": [HumanMessage(content=user_input)]}):
        if 'agent' in chunk and 'messages' in chunk['agent']:
            for message in chunk['agent']['messages']:
                response_result +=message.content
    return response_result