from bs4 import BeautifulSoup
import requests
import os
from langchain_core.messages import HumanMessage,AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain.chains import RetrievalQA

def fetch_webpage_content(url):
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.content, "html.parser")
        title = soup.title.string.strip() if soup.title else "Untitled"
        text = soup.get_text(separator="\n")
        # Limit length if needed
        # short_text = text[:3000]
        return title, text
    except Exception as e:
        return "Error", f"Failed to fetch content: {str(e)}"

def model_feedback(page_text,page_title,user_input):
    user_prompt = f"""
    You are an intelligent AI assistant helping students understand educational content clearly and accurately.

    Below is the content extracted from a reference webpage titled: **{page_title}**.

    Please answer the following question based only on the provided content. If the answer cannot be found in the text, respond politely that the content does not contain the relevant information.

    ---

    ### 📄 Reference Content:
    \"\"\"
    {page_text}
    \"\"\"

    ---

    ### ❓User's Question:
    {user_input}

    ---

    ### 🧠 Instructions:
    - Use only the information in the reference content.
    - Respond clearly, using bullet points or examples if needed.
    - If the content does not answer the question, say so respectfully.
    """

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=os.getenv("GEMINI_API_KEY"),
        temperature=0.5
    )
    tools =[]
    agent_execute = create_react_agent(model=llm,tools=tools)

    response_result = ""
    for chunk in agent_execute.stream({"messages": [HumanMessage(content=user_prompt)]}):
        if "agent" in chunk and "messages" in chunk["agent"]:
            for message in chunk["agent"]["messages"]:
                response_result += message.content

    return response_result
