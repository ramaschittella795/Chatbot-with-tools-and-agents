import streamlit as st
from langchain.agents import initialize_agent, AgentType
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
from langchain.callbacks import StreamlitCallbackHandler
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os
import time

# Load environment variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# LangSmith setup (optional)
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY", "")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "OpenAI Chat Agent with Tools"

# Streamlit Title
st.title("Your Chatbot")

# Tools setup
arxiv_tool = ArxivQueryRun(api_wrapper=ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200))
wiki_tool = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200))
search_tool = DuckDuckGoSearchRun(name="Search")
tools = [search_tool, wiki_tool, arxiv_tool]

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hi! I'm a smart assistant that can search, summarize, and answer your questions. Ask me anything!"}
    ]

# Display chat history
for msg in st.session_state["messages"]:
    st.chat_message(msg["role"]).write(msg["content"])

# Input + Response Handling
if prompt_input := st.chat_input("Ask a question..."):
    if not api_key:
        st.error("OpenAI API Key not found in environment variables.")
    else:
        st.chat_message("user").write(prompt_input)
        st.session_state["messages"].append({"role": "user", "content": prompt_input})

        # LLM setup
        llm = ChatOpenAI(
            model="gpt-4",
            temperature=0.3,
            max_tokens=500,
            openai_api_key=api_key,
            streaming=True
        )

        # Agent with tools
        agent_executor = initialize_agent(
            tools,
            llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            handle_parsing_errors=True
        )

        with st.chat_message("assistant"):
            st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)

            try:
                time.sleep(1)  # Add delay to avoid rate limit
                response = agent_executor.run(prompt_input, callbacks=[st_cb])
            except Exception as e:
                response = f"⚠️ Error: {str(e)}"

            st.session_state["messages"].append({"role": "assistant", "content": response})
            st.write(response)
