# LangChain RAG Project: Unified Custom Data Chat

> An advanced RAG application that builds upon previous LangChain work: [AI Engineering - Study LangChain](https://github.com/beatanemeth/ai-engineering-study-langchain). This project demonstrates **unified indexing** of multiple custom JSON data sources (Wix content) and provides a **stateful, conversational chat** experience constrained to **Hungarian**.

**This project and the engineering experience behind it are discussed in the Medium article**:  
[The AI Engineering Challenge: A Three-Step Journey into RAG, LangChain, and Real-World Data](https://medium.com/@beataspace)

<br></br>

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Project Folder Structure](#2-project-folder-structure)
3. [Architectures Implemented](#3-architectures-implemented)
4. [Technical Stack](#4-technical-stack)
5. [Prerequisites](#5-prerequisites)
6. [Getting Started](#6-getting-started)
7. [Key Architectural Insight](#7-key-architectural-insight)
8. [Resources](#8-resources)
   <br></br>

## 1. Project Overview

This repository contains two distinct LangChain RAG services, each specialized for a specific content retrieval and knowledge management strategy based on a real-world use case:

- **[InsightHubAI](/langchain_rag_services/insight_hub_ai.py)** : Acts as an expert assistant for retrieving data from **archive content**. It uses JSON files from the live website: Blog, Articles, and Events.
- **[ContentNavigatorAI](/langchain_rag_services/content_navigator_ai.py)** : Acts as an expert assistant for **formulating answers** based on a specific knowledge base. It uses JSON files from the live website: Blog and Articles.

### Data Source Note

As the administrator of the [source website](https://www.kiutarakbol.hu/) at the time of project construction, I was able to use Wix Velo code to download the necessary JSON files for personal use.

> **IMPORTANT**:  
> To run this project, you must provide your own custom JSON files and adapt the **`jq_schema`** variables in the code to match your specific data structure.
> <br></br>

## 2. Project Folder Structure

```bash
.
├── assets
├── data                              # Inside the repo only data_example/
│   ├── wix_articles_data.json
│   ├── wix_events_data.json
│   └── wix_posts_data.json
├── embeddings_content_navigator      # Embedding for the ContentNavigatorAI - not included inside the repo
│   └── chroma_persistent_storage
├── embeddings_insight_hub            # Embedding for the InsightHubAI - not included inside the repo
│   └── chroma_persistent_storage
├── .env                              # Inside the repo only .env.example
├── getData
│   ├── jwt_microservice              # FastAPI service for JWT generation
│   │   ├── main.py
│   │   ├── README.md                 # FastAPI/JWT microservice specific README.md file
│   │   ├── requirements.txt
│   │   ├── routers
│   │   │   ├── __init__.py
│   │   │   ├── articles.py
│   │   │   ├── events.py
│   │   │   ├── posts.py
│   │   └── utils
│   │       ├── __init__.py
│   │       ├── data_fetcher.py
│   │       ├── jwt_generator.py
│   └── wixVeloCode                   # Wix Velo backend code
│       ├── http-functions.js
│       ├── README.md
│       ├── services.web.js
│       └── utils.web.js
├── .gitignore
├── langchain_rag_services            # LangChain-RAG services
│   ├── content_navigator_ai.py
│   ├── insight_hub_ai.py
│   ├── README.md                     # LangChain services specific README.md file
│   └── requirements.txt
├── README.md                         # Project wide README.md file
├── requirements.txt
├── utils
│   ├── __init__.py
│   └── logging.py
└── .venv                             # Python Virtual Environment files - not included inside the repo

```

<br></br>

## 3. Architectures Implemented

This repository provides two distinct, executable services built with LangChain & RAG:

| File                      | User Interface                                    | Details                                                                                                    |
| :------------------------ | :------------------------------------------------ | :--------------------------------------------------------------------------------------------------------- |
| `insight_hub_ai.py`       | continuous Q&A session in the CLI                 | Personalized Q&A (Archival): Specialized retrieval from blog, events, and other articles content (Memoir). |
| `content_navigator_ai.py` | continuous Q&A session in the browser (Streamlit) | Unified Knowledge Base (Wiki/FAQ): General Q&A across the combined blog and other articles content.        |

<br></br>

## 4. Technical Stack

| Component            | Detail                                                        | Use                                                                           |
| :------------------- | :------------------------------------------------------------ | :---------------------------------------------------------------------------- |
| **Orchestration**    | **LangChain**                                                 | Manages the sequence of RAG steps, including data ingestion and chat history. |
| **LLM Provider**     | OpenRouter: `google/gemma-3-27b-it:free`                      | Generates Hungarian answers and rewrites historical queries.                  |
| **Embedding Model**  | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | Runs locally to create vector representations.                                |
| **Vector DB**        | [Chroma](https://docs.trychroma.com/) (Persistent)            | Stores and indexes the unified document embeddings.                           |
| **JSON Loader**      | **`JSONLoader`** with **`jq`** Schemas                        | Extracts structured content from heterogeneous JSON files.                    |
| **Source Data**      | [Blog & Tudástár & Events](https://www.kiutarakbol.hu/)       | The proprietary knowledge base for grounding answers.                         |
| **User Interaction** | CLI: **InsightHubAI**; Streamlit: **ContentNavigatorAI**      | Provides a platform for question input.                                       |
| **Development OS**   | Linux Mint 21.2                                               | The system used for development.                                              |

<br></br>

## 5 Prerequisites

You must have the following installed and configured:

- **Python 3.10.12+**
  > **Version Note:** This project was developed and tested using **Python 3.10.12**. While most dependencies will work with newer versions (e.g., Python 3.11/3.12), it is recommended using Python 3.10 or a compatible version to ensure environmental stability.
- An **OpenRouter API Key** (Set as `OPENROUTER_API_KEY` in the `.env` file).

<br></br>

## 6. Getting Started

Following these steps, you will install and run all core components of this project: the data fetching service and both RAG applications.

> **Note:** The setup below assumes you want to **_install all dependencies_** from the root `requirements.txt` file for a full environment setup.

**IMPORTANT:** If you want to run **_only one service_**, please refer to the dedicated instructions in these subdirectories:

- [README file for **FastAPI/JWT microservice**](/getData/jwt_microservice/README.md)
- [README file for **LangChain app**](/langchain_rag_services/README.md)

---

### 6.1. Data Source Requirement

**IMPORTANT:** As noted in the Project Overview, you must provide your own custom JSON files (`wix_posts_data.json`, `wix_articles_data.json`, etc.) in the project's `/data` folder.

You must also adapt the `jq_schema` variables in the LangChain application code to match the structure of your custom data.

---

### 6.2. Configuration (`.env`)

In the root directory of this project, rename the `.env.example` to `.env`.

Populate the file with your environment keys. The required variables for all services are:

#### **LangChain App**

- `OPENROUTER_API_KEY`: Your OpenRouter API key.

#### **FastAPI/JWT Microservice**

- `WIX_AUTH_SECRET`: The shared secret key used for signing JWTs.
- `JWT_SUBJECT_*` values (e.g., `JWT_SUBJECT_EVENTS`): The subject identifiers for each service type.
- `WIX_*_ENDPOINT` values (e.g., `WIX_EVENTS_ENDPOINT`): The full URL for each endpoint you want to call.

**Security Tip:**  
Never commit your `.env` file to version control.

---

### 6.3. Setup Python Virtual Environment

It is best practice to use a virtual environment to isolate project dependencies.

#### **1. Create the environment**

```bash
python3 -m venv .venv
```

#### **2. Activate the environment**

**macOS/Linux:**

```bash
source .venv/bin/activate
```

**Windows (Command Prompt):**

```bash
.venv\Scripts\activate.bat
```

**Windows (PowerShell):**

```bash
.venv\Scripts\Activate.ps1
```

Your command prompt will now show the environment name, like:

```
(.venv) user@host:~/project$
```

indicating that it is active.

---

### 6.4. Update pip

```bash
python -m pip install --upgrade pip
```

---

### 6.5. Install Dependencies

With the virtual environment active, install all necessary packages from the project's root `requirements.txt`:

```bash
pip install -r requirements.txt
```

---

### 6.6. Run the Applications

You will typically run the FastAPI microservice first to fetch the data, and then run the RAG applications to use the data.

#### **1. Start the FastAPI/JWT Microservice (Data Fetcher)**

Change directory to the service location and start the Uvicorn server:

```bash
# Navigate to the service directory
cd getData/jwt_microservice/

# Start the server (runs on http://127.0.0.1:8000 by default)
uvicorn main:app --reload
```

Leave this terminal open.

The service is now running in the background. From a new terminal, you can access endpoints (e.g., `http://127.0.0.1:8000/downloadEvents`) to fetch and populate your local `/data` directory.

#### **2. Run the LangChain RAG Applications**

Open a new terminal window, and run the command:

```bash
# InsightHubAI (CLI)
python3 langchain_rag_services/insight_hub_ai.py

# ContentNavigatorAI (Streamlit)
streamlit run langchain_rag_services/content_navigator_ai.py
```

---

### 6.7. Shut Down and Deactivate

When finished with development or testing, follow these steps to properly shut down all components and exit the environment.

#### **1. Stop the servers**

- **FastAPI/JWT Microservice:** Press `Ctrl + C` in the terminal running `uvicorn`.
- **ContentNavigatorAI (Streamlit):** Press `Ctrl + C` in the terminal running `streamlit`.

#### **2. Deactivate the environment**

```bash
deactivate
```

Your command prompt will return to its default state, and the environment name `(.venv)` will disappear.

<br></br>

## 7. Key Architectural Insight

During development, two RAG applications were evaluated using the same LLM, vector store, and retriever strategies, yet they produced noticeably different results.

- **ContentNavigatorAI** consistently generated more accurate and grounded answers.

- **InsightHubAI** often returned partially accurate responses or explicitly answered “I do not know.”

The critical difference was what was embedded:

**ContentNavigatorAI** indexed rich, descriptive text (article leads, summaries, explanatory content), providing strong semantic signals for retrieval and grounding.

**InsightHubAI** relied on thin, metadata-heavy representations, which proved insufficient for semantic retrieval when answering questions involving aggregation, completeness, or temporal filtering (e.g. “How many events occurred in a given year?”).

This led to a key realization:

> RAG systems are not databases.

Despite experimenting with multiple retriever strategies (Similarity Search, MMR, Self-Querying), none reliably solved questions better suited for structured queries or data aggregation. The issue was not retriever tuning or chunk size, but an architectural mismatch between the problem and the abstraction.
<br></br>

## 8. Resources

[Get started with Chroma vector stor](https://docs.langchain.com/oss/python/integrations/vectorstores/chroma)

[Build a RAG agent with LangChain](https://docs.langchain.com/oss/python/langchain/rag)

[JSONLoader](https://docs.langchain.com/oss/python/integrations/document_loaders/json)

[Sentence Transformer](https://www.sbert.net/docs/sentence_transformer/pretrained_models.html)

[sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2)

[ChromaDB](https://docs.trychroma.com/docs/overview/introduction)

[OpenRouterAi](https://openrouter.ai/)

[Streamlit](https://docs.streamlit.io/)
