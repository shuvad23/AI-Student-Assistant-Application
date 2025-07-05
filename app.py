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
import validators

from create_vectorstore import save_vectorDatabase,load_vectorDatabase,process_add_and_pdfs_to_vectorDB
from rag_response import get_rag_response
from image_response import generate_image_response
from generate_text_response import generate_text
from web_search import fetch_webpage_content,model_feedback

load_dotenv()

if __name__ == "__main__":
    # set streamlit UI
    st.set_page_config(page_title="NeuroNote-AI Student Assistant Application",layout="centered")
    st.markdown("📘 Hey there! 👋 I'm NeuroNote AI, your smart study companion.Let's boost your learning — one note at a time!")


    # Memory for chat and vectorstore
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            AIMessage(content="Hi! I am your NeuroNote-AI . How can i help you ?")
        ]
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = load_vectorDatabase()
    
    # in sidebar (upload pdf's and image option)
    with st.sidebar:
        st.title("📓NeuroNote - AI",width="stretch")
        st.subheader("Your personal AI companion for smarter studying.From notes to notifications — everything in one place.")
        uploaded_pdfs = st.file_uploader("Upload one or more PDFs for RAG context", type=["pdf"],accept_multiple_files=True)
        if uploaded_pdfs:
            st.session_state.vectorstore = process_add_and_pdfs_to_vectorDB(uploaded_pdfs,_existing_vectorstore=st.session_state.vectorstore)
            save_vectorDatabase(st.session_state.vectorstore)
            st.success(f"✅ PDF processed and indexed {len(uploaded_pdfs)} PDF(s)")
            
        # Image input
        uploaded_image = st.file_uploader("Upload image (JPG/PNG)", type=["jpg", "jpeg", "png"])
        agent_types = [
                    "🧠 Super Conscious Agent (All Subjects Expert)",

                    # all core subjects
                    "📘 Math Tutor Agent",
                    "🔢 Algebra & Calculus Assistant",
                    "📐 Geometry & Trigonometry Assistant",
                    "🧮 Statistics & Probability Helper",
                    "🔬 Science Explainer Agent",
                    "🧲 Physics Problem Solver",
                    "🧪 Chemistry Assistant",
                    "🧬 Biology Helper",
                    "📖 Literature & English Agent",
                    "📝 Essay Writing Coach",
                    "🧠 Psychology Study Agent",
                    "🌍 Geography Guide",
                    "📚 History Fact Checker",
                    "⚖️ Civics & Social Studies Agent",
                    "📊 Economics & Business Analyst",
                    "💻 Programming Mentor (Python, C++, Java, etc.)",
                    "🗣️ Language Learning Agent (French, Spanish, etc.)",
                    "🎨 Art & Design Advisor",
                    "🎼 Music Theory Tutor",
                    "🧭 Exam & Revision Planner",
                    
                    # Engineering Section
                    "⚙️ Mechanical Engineering Assistant",
                    "🔌 Electrical Engineering Helper",
                    "🏗️ Civil Engineering Guide",
                    "🖥️ Computer Engineering Mentor",
                    "🧪 Chemical Engineering Tutor",
                    "📡 Electronics & Communication Engineer Agent",
                    "🤖 Robotics & Automation Specialist",
                    "📐 Structural Engineering Consultant",
                    "🌐 Environmental Engineering Advisor",
                    "🚀 Aerospace Engineering Assistant",

                    # Computer Science Section
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
                    
                ]

        # subject select your agent type:
        agent = st.selectbox("Select Your Agent Type:",agent_types)




        # searching by web-link 
        # Initialize session state
        if "active_button" not in st.session_state:
            st.session_state.active_button = None
        web_link = st.text_input("Enter your reference link:")
        active,inactive = st.columns(2)

        page_title, page_text = ("None", "")
        with active:
            if st.button("Active"):
                if web_link and validators.url(web_link):
                    title, text = fetch_webpage_content(web_link)
                    if text:
                        st.session_state.active_button = "on"
                        st.session_state.page_title = title
                        st.session_state.page_text = text
                        st.success("Link successfully connected")
                else:
                    st.error("❌ Invalid URL.")
        with inactive:
            if st.button("Disconnect"):
                st.session_state.active_button = None
                st.success("Link Disconnect")
        
    # display chat history ---
    for msg in st.session_state.chat_history:
        if isinstance(msg,HumanMessage):
            with st.chat_message("user"):
                st.markdown(msg.content)
        else:
            with st.chat_message("assistant"):
                st.markdown(msg.content)



    user_input = st.chat_input("Type your messages....")
    if user_input:
        st.session_state.chat_history.append(HumanMessage(content=user_input))
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message('assistant'):
            with st.spinner("Thinking.."):
                try:
                    if st.session_state.active_button == "on":
                        st.write("📄 Based on your provided web-link:")
                        response = model_feedback(
                                                page_text=st.session_state.get("page_text", ""),
                                                page_title=st.session_state.get("page_title", "Unknown"),
                                                user_input=user_input
                                            )
                    elif st.session_state.vectorstore and uploaded_pdfs:
                        st.write("📄 Based on your uploaded PDF(s):")
                        response = get_rag_response(user_input, st.session_state.vectorstore)
                    elif uploaded_image:
                        st.write("🖼️ Based on your uploaded image:")
                        response = generate_image_response(user_input, uploaded_image)
                    else:
                        response = generate_text(user_input,st.session_state.chat_history,agent)

                    st.markdown(response)
                    st.session_state.chat_history.append(AIMessage(content=response))

                except ValueError:
                    st.error("⚠️ Please provide a valid question or input.")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")