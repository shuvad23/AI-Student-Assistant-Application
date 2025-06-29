from langchain_core.messages import HumanMessage,AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import tool
from langgraph.prebuilt import create_react_agent
import base64
import io
from PIL import Image
import os
from langchain.chains import RetrievalQA

def generate_image_response(prompt, image_file):
    image_bytes = image_file.read()
    image_b64 = base64.b64encode(image_bytes).decode()
    image_data_url = f"data:image/jpeg;base64,{image_b64}"

    vision_llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=os.getenv("GEMINI_API_KEY")
    )

    user_msg = HumanMessage(content=[
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": image_data_url}},
    ])

    response = vision_llm.invoke([user_msg])
    return response.content