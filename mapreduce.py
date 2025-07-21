import validators
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.chains.llm import LLMChain
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader
from langchain_groq import ChatGroq

# Streamlit APP setup
st.set_page_config(page_title="LangChain: Multilingual Summary to English", page_icon="🌍")
st.title("🌍 LangChain: Multilingual Summary to English")
st.subheader('Summarize Any URL (Website or YouTube) in English')

# Sidebar for Groq API Key
with st.sidebar:
    groq_api_key = st.text_input("Groq API Key", value="", type="password")

# URL Input
generic_url = st.text_input("Paste the YouTube or Website URL here", label_visibility="collapsed")

# Map Prompt Template - Multilingual Translation and Summarization
map_prompt_template = """You will be given text in any language.
First, translate it into English.
Then, write a concise English summary of the following content:
{text}
"""
map_prompt = PromptTemplate(template=map_prompt_template, input_variables=["text"])

# Combine Prompt Template
combine_prompt_template = """You will be given multiple English summaries.
Combine them into a single cohesive English summary in under 300 words:
{text}
"""
combine_prompt = PromptTemplate(template=combine_prompt_template, input_variables=["text"])

# Button click
if st.button("Summarize the Content from YT or Website"):

    if not groq_api_key.strip() or not generic_url.strip():
        st.error("Please provide both the Groq API key and a URL.")
    elif not validators.url(generic_url):
        st.error("Please enter a valid URL. It can be a YouTube or website URL.")
    else:
        try:
            with st.spinner("Loading and summarizing..."):
                # Load the Groq LLM
                llm = ChatGroq(model="llama3-70b-8192", groq_api_key=groq_api_key)

                # Load YouTube transcript
                if "youtube.com/watch?v=" in generic_url or "youtu.be/" in generic_url:
                    try:
                        loader = YoutubeLoader.from_youtube_url(generic_url)
                        docs = loader.load()
                        if not docs:
                            st.error("No transcript found. The video may not have captions or might be restricted.")
                            st.stop()
                    except Exception as yt_error:
                        st.error(f"Failed to load YouTube content. Error: {yt_error}")
                        st.stop()
                elif "youtube.com" in generic_url:
                    st.error("Please provide a direct YouTube video link (not Shorts, search, or playlist).")
                    st.stop()
                else:
                    # Load website content
                    try:
                        loader = UnstructuredURLLoader(
                            urls=[generic_url],
                            ssl_verify=False,
                            headers={
                                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36"
                            }
                        )
                        docs = loader.load()
                    except Exception as web_error:
                        st.error(f"Failed to load website content. Error: {web_error}")
                        st.stop()

                # Run the map-reduce summarization with English translation built into prompt
                chain = load_summarize_chain(
                    llm,
                    chain_type="map_reduce",
                    map_prompt=map_prompt,
                    combine_prompt=combine_prompt
                )

                output_summary = chain.run(docs)

                st.success("✅ English summary generated successfully!")
                st.write(output_summary)

        except Exception as e:
            st.exception(f"Exception: {e}")
