
import os
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.tools import tool

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vector_store = Chroma(
    persist_directory="./gemma_memory",
    embedding_function=embedding_model,
    collection_name="web_cache"
)

base_search = DuckDuckGoSearchResults(api_wrapper=DuckDuckGoSearchAPIWrapper(max_results=3))


@tool(description="Searches the internet and automatically embeds the results into the local Vector DB.")
def search_and_memorize(query: str) -> str:
    search_results = base_search.invoke({"query": query})

    doc = Document(page_content=search_results, metadata={"source": "web_search", "query": query})
    vector_store.add_documents([doc])

    return f"Live Web Results:\n{search_results}"


tools = [search_and_memorize]



