from langchain_core.messages import HumanMessage,AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
import os
import requests
from bs4 import BeautifulSoup
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.tools import ArxivQueryRun
from langchain_community.utilities.arxiv import ArxivAPIWrapper
load_dotenv()

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")


def get_tools_for_agent(agent_type):
    tavily_search = TavilySearchResults(k=3)
    arxiv_search = ArxivQueryRun(api_wrapper=ArxivAPIWrapper())
    # Super agent gets everything
    if agent_type == "🧠 Super Conscious Agent (All Subjects Expert)":
        return [tavily_search,arxiv_search]

    # Core Subjects
    if agent_type in [
        "🔬 Science Explainer Agent",
        "🧲 Physics Problem Solver",
        "🧪 Chemistry Assistant",
        "🧬 Biology Helper",
        "📊 Economics & Business Analyst",
        "🧠 Psychology Study Agent",
        "📚 History Fact Checker",
        "⚖️ Civics & Social Studies Agent",
        "🌍 Geography Guide"
    ]:
        return [tavily_search]

    # Engineering Section
    if agent_type in [
        "⚙️ Mechanical Engineering Assistant",
        "🔌 Electrical Engineering Helper",
        "🏗️ Civil Engineering Guide",
        "🖥️ Computer Engineering Mentor",
        "🧪 Chemical Engineering Tutor",
        "📡 Electronics & Communication Engineer Agent",
        "🤖 Robotics & Automation Specialist",
        "📐 Structural Engineering Consultant",
        "🌐 Environmental Engineering Advisor",
        "🚀 Aerospace Engineering Assistant"
    ]:
        return [tavily_search]

    # Computer Science Section
    if agent_type in [
        "💻 Computer Science Researcher",
        "🖥️ Software Development Mentor",
        "🧑‍💻 Algorithms & Data Structures Tutor",
        "🔐 Cybersecurity Advisor",
        "☁️ Cloud Computing Assistant",
        "🤖 Artificial Intelligence Specialist",
        "📊 Data Science Analyst",
        "🌐 Web Development Guide",
        "📱 Mobile App Development Tutor",
        "🧬 Machine Learning Engineer",
        "🛠️ DevOps & Automation Consultant",
        "🔎 Computer Vision Expert",
        "🎮 Game Development Mentor"
    ]:
        return [tavily_search,arxiv_search]

    # Optional: add for these if needed
    if agent_type in [
        "📘 Math Tutor Agent",
        "🔢 Algebra & Calculus Assistant",
        "📐 Geometry & Trigonometry Assistant",
        "🧮 Statistics & Probability Helper"
    ]:
        return [tavily_search]  # Or replace with solve_math_expression if you want real math solving

    # Default (no tool needed)
    return []

def generate_text(user_input,_chat_history_text,agent_type):


    # Directly format the full prompt with user question
    user_input_prompt = f"""
                    You are NeuroNote AI — a smart, friendly, and expert multi-subject assistant 🤖📚, 
                    here to support students with learning, note-taking, and subject-specific guidance.

                    ### Behavior Rules:
                    - Be helpful, clear, and friendly.
                    - Use bullet points, examples, or code where appropriate.
                    - If the user asks something outside your scope:
                        - Politely say it's out of scope
                        - Suggest which agent should handle it (based on the topic)
                        - Example response:
                        > "I'm currently your {agent_type}, so I focus on that subject. But this looks more like a Physics question — would you like me to switch to the 🧲 Physics Problem Solver Agent?"


                    ### Role:
                    You are currently acting as the **{agent_type}**, so respond with knowledge, tone, 
                    and examples suitable for that role. Be clear, concise, and student-friendly.
                    and maybe drop a clever joke here and there. You're that cool teacher everyone loves.

                    ### Personality:
                    - Encouraging, friendly, and non-judgmental
                    - Gives examples where helpful
                    - Explains complex topics in a simple way
                    ### reference link:(if available):
                        Use this content in your response if it's relevant.
                        - Note: 
                                - If a reliable reference link is provided in the input (such as a website or article), use the content from that source to answer the question.
                                - if you not found user query on this provided link or user query and provided link article are not match then reply the context not match for your query
                                - Be sure to mention the title of the webpage or article in your answer for better context. 
                                - If no reference is provided or the query does not directly match existing information, use your own knowledge and reasoning to generate a helpful, accurate, and student-friendly response.
                                
                                
                                
                    ### Conversation So Far:
                    {_chat_history_text}

                    ### New User Question:
                    {user_input}

                    ### Instructions:
                    Respond in a way that fits your selected agent type (**{agent_type}**), while being helpful, accurate, and easy to understand. Use bullet points, math formatting, or code blocks when appropriate.
                        - If the question requires external knowledge (like real-time search, news, or facts),
                          use the available tools like `tavily_search` instead of answering from memory.
                    ### reference link(if available):
                        - give some top most source link for better research on this user query (highlight this link)
                    """
    
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=os.getenv("GEMINI_API_KEY"),
        temperature=0.5
    )
    tools = get_tools_for_agent(agent_type)
    agent_execute = create_react_agent(model=llm,tools=tools)

    response_result = ""
    for chunk in agent_execute.stream({"messages": [HumanMessage(content=user_input_prompt)]}):
        if "agent" in chunk and "messages" in chunk["agent"]:
            for message in chunk["agent"]["messages"]:
                response_result += message.content

    return response_result
