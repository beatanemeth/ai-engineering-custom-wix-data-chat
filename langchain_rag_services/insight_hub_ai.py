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
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# To use a regular import statement for logging utilities you have to ensure that the utils package is discoverable.
from utils import log_info, log_warn, log_error, log_step

# ---*---*---*---*---*---*---*---*---*---*---*---*---*---*---*---
# ⚙️ CONFIGURATIONS
# ---*---*---*---*---*---*---*---*---*---*---*---*---*---*---*---
JSON_POSTS_FILE_PATH = PROJECT_ROOT / "data" / "wix_posts_data.json"
JSON_ARTICLES_FILE_PATH = PROJECT_ROOT / "data" / "wix_articles_data.json"
JSON_EVENTS_FILE_PATH = PROJECT_ROOT / "data" / "wix_events_data.json"

CHROMA_STORAGE_PATH = (
    PROJECT_ROOT / "embeddings_insight_hub" / "chroma_persistent_storage"
)
COLLECTION_NAME = "wix_unified_rag_collection"

OPENROUTER_MODEL = "google/gemma-3-27b-it:free"
EMBEDDINGS_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

load_dotenv()
openrouter_api_key = os.getenv("OPENROUTER_API_KEY")

if not openrouter_api_key:
    log_error("Missing OPENROUTER_API_KEY in environment.")
    sys.exit(1)


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
    model_name=EMBEDDINGS_MODEL, model_kwargs={"device": "cpu"}
)

# 3. Vector Store Check and Indexing
if os.path.exists(CHROMA_STORAGE_PATH):
    log_info(
        f"Vector Store already exists at {CHROMA_STORAGE_PATH}. Loading persistent store."
    )
    vector_store = Chroma(
        persist_directory=str(CHROMA_STORAGE_PATH),
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
    )
    chroma_store_exists = True
else:
    log_info("No persisted Chroma DB found. Will be created in Indexing Phase.")
    vector_store = None
    chroma_store_exists = False


# ---*---*---*---*---*---*---*---*---*---*---*---*---*---*---*---
# 📄 RAG STEP 1 — INDEXING PHASE (PREPARATION / AUGMENTATION)
# Load → Chunk → Embed (Vectorize) → Store
# ---*---*---*---*---*---*---*---*---*---*---*---*---*---*---*---

# -----------------------------------------------------
# JQ Schemas and Metadata (Data Extraction Definition)
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

EVENTS_JQ_SCHEMA = """
.[] | select(type == "object") | {
    # 1. Page Content (Searchable/Vectorized)
    content_key: "TITLE: \(.title) | Published: \((.dateAndTimeSettings.startDate // "") | split(\"T\")[0]) | URL: \(.eventPageUrl)",
    
    # 2. Metadata (Filterable/Searchable Fields)
    filter_title_key: .title,
    filter_date_key: ((.dateAndTimeSettings.startDate // "") | split(\"T\")[0]) ,
    filter_type_key: "esemény"
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
    "wix_events": "Esemény",
}


# Helper function to load and prepare documents.
def load_and_prepare_documents(file_path, jq_schema, source_name):
    """
    Loads documents from a single JSON file, applies the JQ schema for extraction,
    and adds a source_file metadata tag.
    """
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
        # Add a critical metadata tag to distinguish the original JSON source file
        # The source_file metadata ("wix_posts_data.json", etc.) is essential for debugging and for the LLM's instructions.
        source_type_name = SOURCE_TYPE_MAP.get(source_name, "Ismeretlen forrás")
        doc.metadata["source_file"] = source_type_name

    return documents


if not chroma_store_exists:

    log_step("--- 📄 Indexing Phase (RAG Step-1) ---")
    all_documents = []

    # -----------------------------------------------------
    # 1. Load: Iterate and Load All Data Sources
    #    The data is loaded and structured using the JQ Schemas defined above.
    # -----------------------------------------------------

    # Process Wix Posts Data
    all_documents.extend(
        load_and_prepare_documents(JSON_POSTS_FILE_PATH, POSTS_JQ_SCHEMA, "wix_posts")
    )

    # Process Wix Articles Data
    all_documents.extend(
        load_and_prepare_documents(
            JSON_ARTICLES_FILE_PATH, ARTICLES_JQ_SCHEMA, "wix_articles"
        )
    )

    # Process Wix Events Data
    all_documents.extend(
        load_and_prepare_documents(
            JSON_EVENTS_FILE_PATH, EVENTS_JQ_SCHEMA, "wix_events"
        )
    )

    if not all_documents:
        log_error("No documents loaded from any source. Exiting.")
        sys.exit(1)

    log_info(f"Total documents loaded: {len(all_documents)}.")

    # -----------------------------------------------------
    # 2. Chunk: Split Combined Documents
    #    The single list of documents is now broken down into smaller,
    #    context-preserving chunks for better vector search granularity.
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

TOP_K_CHUNKS = 6  # Retrieve the top 6 chunks for context

retriever = vector_store.as_retriever(
    search_type="mmr",  # Maximum Marginal Relevance for diverse results
    search_kwargs={
        "k": TOP_K_CHUNKS,
        "fetch_k": 20,  # Fetch more documents initially for better MMR re-ranking
    },
)

# ---*---*---*---*---*---*---*---*---*---*---*---*---*---*---*---
# 💡 RAG STEP 3 — GENERATION PHASE
# ---*---*---*---*---*---*---*---*---*---*---*---*---*---*---*---
log_step("--- 💡 Generation Phase (RAG Step-3) ---")

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
**Role:You are an **Expert Archivist and Content Researcher**, specializing in managing and querying the entire content database (Events, Posts, and Articles) of 'kiutarakbol.hu'. You are the definitive authority on past happenings, content details, and the ability to count and list specific content types (event, post, article). Treat yourself as a reliable internal database for a library containing every detai
**Goal:Answer the user's question accurately, factually, and concisely, relying STRICTLY and ONLY on the provided context.

**Context:**
The The context below contains relevant chunks of Wix content data (events, posts, and articles) that were retrieved based on the user's current or rewritten query.
--- Retrieved Context ---
{context}
------------------------

**Instructions:**
1.  **Language:** All responses **MUST** be in **Hungarian**.
2.  **Focus:** Base your entire answer *strictly* on the `Retrieved Context`. Focus on titles and content summaries.
3.  **Source Mention:** If relevant, always mention whether the information came from an Event, a Post, or an Article (based on the source_file metadata).
4.  **Formatting:** 
        - When listing multiple pieces of content or structuring requested data, use a clear, numbered, or bulleted list format.
        - For counting tasks (e.g., how many posts are there), provide a simple numerical answer first, followed by an explanation of what was counted.
5.  **Guardrail (Critical):** If the answer to the user's question is not fully present or cannot be definitively inferred from the `Retrieved Context`, you MUST use the following exact Hungarian phrase:
    'Nem találom ezt az információt a megadott tartalom-adatokban.' 

**Chat History for Continuity:**
{chat_history}
""",
        ),
        ("human", "{input}"),
    ]
)


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


# Get the final executable RAG chain
conversational_rag_chain = get_rag_chain(llm_model, retriever)

# Define the final runnable with message history
chain = RunnableWithMessageHistory(
    conversational_rag_chain,
    lambda session_id: ChatMessageHistory(session_id=session_id),
    input_messages_key="input",
    history_messages_key="chat_history",
)

# ---*---*---*---*---*---*---*---*---*---*---*---*---*---*---*---
# ⚙️ QUERY EXECUTION
# ---*---*---*---*---*---*---*---*---*---*---*---*---*---*---*---
SESSION_ID = "content_session"


def execute_query(chain, query, session_id):
    log_step("--- ⚙️ Query Execution ---")
    print(f"QUESTION: {query}")

    result = chain.invoke(
        {"input": query}, config={"configurable": {"session_id": session_id}}
    )

    answer = result["answer"]
    sources = result["context"]

    print(f"ANSWER: {answer}")
    log_step(f"Sources Used ({len(sources)} Chunks):")
    for i, doc in enumerate(sources):
        title = doc.metadata.get("filter_title_key", "N/A (Missing Title)")
        source_type = doc.metadata.get("source_file", "UNKNOWN")

        print(f"Source_{i+1} (Title: {title}, Type: {source_type}):")
        print(f"  {doc.page_content[:200]}...")
        print("-" * 20)


# ---*---*---*---*---*---*---*---*---*---*---*---*---*---*---*---
# CLI Support (Continuous Chat & Termination)
# ---*---*---*---*---*---*---*---*---*---*---*---*---*---*---*---
if __name__ == "__main__":

    if len(sys.argv) > 1:
        user_query = " ".join(sys.argv[1:])
        print(f"\n--- Running Single Command-Line Query ---")
        execute_query(chain, user_query, SESSION_ID)

    print("\n--- 🎉 Wix Content Chat Initiated ---")
    print(f"Session ID: {SESSION_ID}")
    print(
        "Ask a question about the posts or articles or events (e.g., 'What is the title of the latest post?')."
    )
    print("Type 'quit', 'exit', or 'no' to end the chat.")
    print("-" * 40)

    while True:
        try:
            user_input = input("USER > ")

            if user_input.lower() in ["quit", "exit", "no"]:
                print("--- Chat Session Ended. Goodbye! 👋 ---")
                break

            if not user_input.strip():
                continue

            execute_query(chain, user_input, SESSION_ID)

        except KeyboardInterrupt:
            print("\n--- Chat Session Ended by User (Ctrl+C). Goodbye! 👋 ---")
            break
        except Exception as e:
            log_error(f"An unexpected error occurred: {e}")
            break
