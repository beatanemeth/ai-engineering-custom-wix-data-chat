import streamlit as st
import os
from dotenv import load_dotenv
import sys
from pathlib import Path

# Core Abstractions and Base Classes
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory

# LLM and Embedding Integrations
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import SentenceTransformerEmbeddings

# Community/Third-Party Components
from langchain_community.document_loaders import JSONLoader
from langchain_community.vectorstores import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory

# High-Level Chains (simplified imports for standard RAG components)
from langchain_classic.chains import (
    create_history_aware_retriever,
    create_retrieval_chain,
)
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

# Dedicated Splitters Package
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Define the base directory (the project root)
try:
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
except NameError:
    # Fallback if __file__ is not defined
    PROJECT_ROOT = Path(".").resolve().parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


from utils import log_info, log_warn, log_error, log_step


# ---*---*---*---*---*---*---*---*---*---*---*---*---*---*---*---
# ⚙️ CONFIGURATIONS
# ---*---*---*---*---*---*---*---*---*---*---*---*---*---*---*---
JSON_POSTS_FILE_PATH = PROJECT_ROOT / "data" / "wix_posts_data.json"
JSON_ARTICLES_FILE_PATH = PROJECT_ROOT / "data" / "wix_articles_data.json"

CHROMA_STORAGE_PATH = (
    PROJECT_ROOT / "embeddings_content_navigator" / "chroma_persistent_storage"
)
COLLECTION_NAME = "wix_unified_rag_collection"

OPENROUTER_MODEL = "google/gemma-3-27b-it:free"
EMBEDDINGS_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

TOP_K_CHUNKS = 6
SESSION_ID = "content_session"

load_dotenv()
openrouter_api_key = os.getenv("OPENROUTER_API_KEY")

# -----------------------------------------------------
# JQ SCHEMA DEFINITIONS FOR JSON LOADING
#    These schemas define how to map raw JSON objects into
#    LangChain Document Page Content (for vectorization)
#    and Metadata (for filtering).
# -----------------------------------------------------

POSTS_JQ_SCHEMA = """
.[] | select(type == "object") | {
    # 1. Page Content (Searchable/Vectorized)
    content_key: "TITLE: \(.title) | Excerpt: \(.excerpt) | Published: \(.firstPublishedDate | split(\"T\")[0]) | URL: \(.url) | Views: \(.metrics.views) | Likes: \(.metrics.likes)",
    
    # 2. Metadata (Filterable/Searchable Fields)
    filter_content_key: .excerpt,
    filter_title_key: .title,
    filter_date_key: (.firstPublishedDate | split("T")[0]),
    filter_type_key: "poszt"
}
"""

ARTICLES_JQ_SCHEMA = """
.items[] | select(type == "object") | {
    # 1. Page Content (Searchable/Vectorized)
    content_key: "TITLE: \(.title) | Published: \((.publishDate // "") | split(\"T\")[0]) | URL: https://www.kiutarakbol.hu/\(.urlvege1)/\(.urlvege)",    
    
    # 2. Metadata (Filterable/Searchable Fields)
    filter_content_key: .lead,
    filter_title_key: .title,
    filter_date_key: ((.publishDate // "") | split(\"T\")[0]),
    filter_type_key: "tudástár"
}
"""


# Helper function defining the metadata extraction function.
# The metadata_func is responsible for identifying which pieces of information in the record should be included in the metadata stored in the final Document object.
def metadata_func(record: dict, metadata: dict) -> dict:
    metadata["filter_content_key"] = record.get("filter_content_key", "N/A")
    metadata["filter_title_key"] = record.get("filter_title_key", "N/A")
    metadata["filter_date_key"] = record.get("filter_date_key", "N/A")
    metadata["filter_type_key"] = record.get("filter_type_key", "N/A")
    return metadata


# Helper function to map file names to user-friendly source types (in Hungarian)
SOURCE_TYPE_MAP = {
    "wix_posts": "Blog poszt",
    "wix_articles": "Tudástár cikk",
}


# Helper function to load and prepare documents.
def load_and_prepare_documents(file_path, jq_schema, source_name):
    if not os.path.exists(file_path):
        log_warn(f"JSON not found: {file_path}. Skipping.")
        return []

    log_info(f"Loading document from {file_path} (Source: {source_name})...")

    loader = JSONLoader(
        file_path=str(file_path),
        jq_schema=jq_schema,
        content_key="content_key",
        metadata_func=metadata_func,
    )

    documents = loader.load()
    log_info(f"Loaded {len(documents)} objects from {source_name}.")

    for doc in documents:
        source_type_name = SOURCE_TYPE_MAP.get(source_name, "Ismeretlen forrás")
        doc.metadata["source_file"] = source_type_name

    return documents


# ------------------------------------------------------
# PROMPT DEFINITIONS
# ------------------------------------------------------
# 1. Contextualizing Question Prompt (for Query Rewriting)
CONTEXTUALIZE_Q_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Given a chat history and the latest user question, generate a standalone question that can be used to search the vector store. The input and output query must be in **Hungarian**. Do not answer the question, just rephrase it if necessary. If no history exists, return the question as is.",
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ]
)

# 2. Final Answer Prompt (Highly Structured for Generation)
FINAL_ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
**Role:** You are an **Expert Wix Content Assistant** specialized in providing factual information from the 'kiutarakbol.hu' posts and articles.
**Goal:** Answer the user's question accurately and concisely, drawing ONLY from the provided context.

**Context:**
The context below contains relevant chunks of Wix content data (posts and articles) that were retrieved based on the user's current or rewritten query.
--- Retrieved Context ---
{context}
------------------------

**Instructions:**
1.  **Language:** All responses **MUST** be in **Hungarian** (Magyarul).
2.  **Focus:** Base your entire answer *strictly* on the `Retrieved Context`. Focus on titles and content summaries.
3.  **Source Mention:** If relevant, you can mention whether the information came from a **Post** or an **Article** (based on the `source_file` metadata).
4.  **Formatting:** Use a clear list format (bullet points or numbered list) when listing multiple pieces of content.
5.  **Guardrail (Critical):** If the answer to the user's question is not fully present or cannot be definitively inferred from the `Retrieved Context`, you MUST use the following exact Hungarian phrase:
    'Nem találom ezt az információt a megadott tartalom-adatokban.' 

**Chat History for Continuity:**
{chat_history}
""",
        ),
        ("human", "{input}"),
    ]
)


# ------------------------------------------------------
# CHAIN UTILITIES
# ------------------------------------------------------
def get_history_aware_retriever(llm_model, retriever):
    """
    Creates a retriever chain that reformulates follow-up questions
    into standalone search queries using the chat history.
    """
    history_aware_retriever = create_history_aware_retriever(
        llm_model, retriever, CONTEXTUALIZE_Q_PROMPT
    )
    return history_aware_retriever


def get_rag_chain(llm_model, retriever):
    """
    Creates the complete Conversational RAG chain by combining
    the history-aware retriever with the final generation chain.
    """
    # 1. Create the History-Aware Retriever (Query Rewriting)
    history_aware_retriever = get_history_aware_retriever(llm_model, retriever)

    # 2. Create the Document Combiner Chain (Stuffing and Generation)
    combine_docs_chain = create_stuff_documents_chain(llm_model, FINAL_ANSWER_PROMPT)

    # 3. Create the Final Retrieval Chain (Orchestration)
    rag_chain = create_retrieval_chain(history_aware_retriever, combine_docs_chain)

    return rag_chain


# ---*---*---*---*---*---*---*---*---*---*---*---*---*---*---*---
# 💻 STREAMLIT INTEGRATION
# ---*---*---*---*---*---*---*---*---*---*---*---*---*---*---*---

# --- CONFIGURATION & PAGE SETUP ---
st.set_page_config(layout="wide")

# --- CUSTOM CSS INJECTION ---
st.markdown(
    """
    <style>
    /* Sidebar Width */
    section[data-testid="stSidebar"] {
        width: 450px !important; 
    }
    
    /* Hide the default sidebar expander arrow for a cleaner look */
    .st-emotion-cache-19p0t4a {
        visibility: hidden;
    }

    .blue-separator-line {
        height: 2px; /* Thickness of the line */
        background-color: #4a90e2; /* Deep blue color */
        margin: 20px 0; /* Vertical spacing */
        width: 100%; /* Ensure it spans the full width */
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# Initialize Chat Message History in session state
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []

# Initialize the history store for LangChain
# NOTE: Streamlit's cache needs a *history factory* for LangChain
# We will use a dictionary as a simple in-memory store for session histories
# Key is the session_id (e.g., "content_session"), value is the ChatMessageHistory object.
history_store = {}


def get_session_history(session_id: str) -> ChatMessageHistory:
    if session_id not in history_store:
        history_store[session_id] = ChatMessageHistory()
    return history_store[session_id]


# ------------------------------------------------------
# LangChain & RAG Setup
# ------------------------------------------------------
@st.cache_resource
def setup_rag_system():

    if not openrouter_api_key:
        log_error("Missing OPENROUTER_API_KEY in environment. Exiting setup.")
        # Raise an error or return None, Streamlit will catch it
        st.error("RAG Setup Failed: Missing OPENROUTER_API_KEY in environment.")
        return None

    # ---*---*---*---*---*---*---*---*---*---*---*---*---*---*---*---
    # 🧩 LangChain COMPONENTS INITIALIZATION
    # ---*---*---*---*---*---*---*---*---*---*---*---*---*---*---*---
    log_step("--- 🧩 Initializing LangChain Components ---")

    # 1. LLM Model Initialization
    llm_model = ChatOpenAI(
        model=OPENROUTER_MODEL,
        api_key=openrouter_api_key,
        base_url="https://openrouter.ai/api/v1",
        temperature=0,
    )

    # 2. Embeddings Model Initialization
    embeddings = SentenceTransformerEmbeddings(
        model_name=EMBEDDINGS_MODEL,
        model_kwargs={"device": "cpu"},
    )

    # 3. Vector Store Check and Indexing
    chroma_store_exists = os.path.exists(CHROMA_STORAGE_PATH)

    if chroma_store_exists:
        log_info(
            f"Vector Store already exists at {CHROMA_STORAGE_PATH}. Loading persistent store."
        )
        vector_store = Chroma(
            persist_directory=str(CHROMA_STORAGE_PATH),
            embedding_function=embeddings,
            collection_name=COLLECTION_NAME,
        )
    else:
        # ---*---*---*---*---*---*---*---*---*---*---*---*---*---*---*---
        # 📄 RAG STEP 1 — INDEXING PHASE (PREPARATION / AUGMENTATION)
        # Load → Chunk → Embed (Vectorize) → Store
        # ---*---*---*---*---*---*---*---*---*---*---*---*---*---*---*---
        log_step("--- 📄 Indexing Phase (RAG Step-1) ---")
        all_documents = []

        # -----------------------------------------------------
        # 1. Load: Iterate and Load All Data Sources
        #    The data is loaded and structured using the JQ Schemas defined above.
        # -----------------------------------------------------

        # Process Wix Posts Data
        all_documents.extend(
            load_and_prepare_documents(
                JSON_POSTS_FILE_PATH, POSTS_JQ_SCHEMA, "wix_posts"
            )
        )

        # Process Wix Articles Data
        all_documents.extend(
            load_and_prepare_documents(
                JSON_ARTICLES_FILE_PATH, ARTICLES_JQ_SCHEMA, "wix_articles"
            )
        )

        if not all_documents:
            log_error("No documents loaded from any source. RAG setup failed.")
            st.error("RAG Setup Failed: Nincsenek betöltött dokumentumok.")
            return None

        log_info(f"Total documents loaded: {len(all_documents)}.")

        # -----------------------------------------------------
        # 2. Chunk: Split Combined Documents
        #    The single list of documents is now broken down into smaller,
        #    context-preserving chunks for better vector search gr
        # -----------------------------------------------------
        log_info("2. Splitting documents into chunks...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
        )
        chunked_docs = text_splitter.split_documents(all_documents)
        log_info(f"Created {len(chunked_docs)} chunks for indexing.")

        # -----------------------------------------------------
        # 3. Embed & 4. Store: Create and Persist Vector Store
        #    The chunked documents are vectorized using the embedding model
        #    and saved to the local ChromaDB persistent storage.
        # -----------------------------------------------------
        log_info(
            f"3. & 4. Creating and persisting Chroma Vector Store at {CHROMA_STORAGE_PATH}..."
        )
        vector_store = Chroma.from_documents(
            documents=chunked_docs,
            embedding=embeddings,
            persist_directory=str(CHROMA_STORAGE_PATH),
            collection_name=COLLECTION_NAME,
        )
        vector_store.persist()

    # ---*---*---*---*---*---*---*---*---*---*---*---*---*---*---*---
    # 🔍 RAG STEP 2 — RETRIEVAL PHASE
    # ---*---*---*---*---*---*---*---*---*---*---*---*---*---*---*---
    log_step("--- 🔍 Retrieval Phase (RAG Step-2) ---")
    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": TOP_K_CHUNKS,
            "fetch_k": 20,
        },
    )

    # ---*---*---*---*---*---*---*---*---*---*---*---*---*---*---*---
    # 💡 RAG STEP 3 — GENERATION PHASE
    # ---*---*---*---*---*---*---*---*---*---*---*---*---*---*---*---
    # Generation Chain Assembly
    log_step("--- 💡 Generation Phase (RAG Step-3) ---")
    conversational_rag_chain = get_rag_chain(llm_model, retriever)

    # Final RunnableWithMessageHistory
    final_conversational_chain = RunnableWithMessageHistory(
        conversational_rag_chain,
        get_session_history,  # Use the defined function as the history factory
        input_messages_key="input",
        history_messages_key="chat_history",
    )

    st.success("Készen állok a kérdéseidre!")
    return final_conversational_chain


# Execute the cached setup function
final_runnable = setup_rag_system()

# --- SIDEBAR ---
with st.sidebar:

    st.header("📚 Tudástár AI")

    st.markdown("---")
    st.header("ℹ️ Rólunk")
    st.info(
        f"""
        Ez egy mesterséges intelligencia által vezérelt, beszélgetőképes keresőeszköz, 
        amely a 'kiutarakbol.hu' tartalmi archívumából (Posztok és Cikkek) 
        gyűjt össze információkat.

        **Modell:** {OPENROUTER_MODEL.split('/')[-2]} (OpenRouter-en keresztül)
        **Visszakeresés:** MMR-alapú RAG (k={TOP_K_CHUNKS})
    """
    )
    st.markdown("---")

    if st.button("Beszélgetési előzmények törlése", type="primary"):
        st.session_state.chat_messages = []
        # Clear the LangChain history store as well
        if SESSION_ID in history_store:
            del history_store[SESSION_ID]
        st.rerun()

# --- MAIN CONTENT AREA ---
st.header("Tartalomkutató Chat 🤖")

st.markdown("-------------------------")

# Display previous messages from session state
for message in st.session_state.chat_messages:
    with st.chat_message(message["role"]):
        if message["role"] == "user":
            st.markdown(message["content"])
        else:
            # Display the AI answer text
            st.markdown(message["content"]["text"])

            # Display persistent Source Panel for past AI answers
            sources = message["content"]["sources"]
            with st.expander(f"📚 Felhasznált források ({len(sources)} db)"):
                for i, doc in enumerate(sources):
                    title = doc.metadata.get("filter_title_key", "N/A")
                    source_type = doc.metadata.get("source_file", "ISMERETLEN")

                    st.markdown(f"**{i+1}. {title}** (Típus: *{source_type}*)")
                    st.code(
                        doc.page_content.strip()[:200] + "...", language="text"
                    )  # Use up to 200 chars for better context

            # Separator Line before new input
            if len(st.session_state.chat_messages) >= 1:
                st.markdown(
                    '<div class="blue-separator-line"></div>', unsafe_allow_html=True
                )

# Handle user input via the dedicated chat input widget
if prompt := st.chat_input(
    "Tegyél fel egy kérdést a kiutarakbol.hu tartalmával kapcsolatban..."
):

    if final_runnable is None:
        st.error(
            "A RAG rendszer inicializálása sikertelen. Ellenőrizd a környezeti változókat és a fájl elérési utakat."
        )
        st.stop()

    # 1. Store and display user question
    st.session_state.chat_messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Invoke RAG chain and get result
    with st.chat_message("assistant"):
        with st.spinner("Gondolkodom..."):

            # Execute the RAG chain
            result = final_runnable.invoke(
                {"input": prompt},
                config={
                    "configurable": {"session_id": SESSION_ID}
                },  # Use the predefined session ID
            )

        answer = result["answer"]
        sources = result["context"]

        # 3. Store the full structured AI Answer
        ai_message_content = {"text": answer, "sources": sources}
        st.session_state.chat_messages.append(
            {"role": "assistant", "content": ai_message_content}
        )

        # Display AI Answer
        st.markdown(answer)

        # 4. Source Panel (Attribution) - Display for the current answer
        with st.expander(f"📚 Felhasznált források ({len(sources)} db)"):
            for i, doc in enumerate(sources):
                title = doc.metadata.get("filter_title_key", "N/A")
                source_type = doc.metadata.get("source_file", "ISMERETLEN")

                st.markdown(f"**{i+1}. {title}** (Típus: *{source_type}*)")
                # Show up to 200 characters of the content
                st.code(doc.page_content.strip()[:200] + "...", language="text")

    # Force rerun to display the newly added assistant message immediately
    st.rerun()
