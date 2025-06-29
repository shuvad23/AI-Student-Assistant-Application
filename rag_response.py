import os
from langchain_core.messages import HumanMessage,AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
load_dotenv()
def get_rag_response(user_query, vectorstore):
    retriever = vectorstore.as_retriever(search_type="similarity", k=3)

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=os.getenv("GEMINI_API_KEY"),
        temperature=0.4
    )

    rag_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=False)
    result = rag_chain.invoke({"query": user_query})
    return result["result"]
