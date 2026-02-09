# LangChain Apps

> In case, that you only would like to use this service, follow the steps bellow for the installation.

## Getting Started 🚀

Using a virtual environment (`.venv`) is the best practice for Python projects. It isolates your project's dependencies from your system-wide Python installation, preventing conflicts.

Here is the step-by-step process for setting up your environment and installing dependencies:

## Getting Started 🚀

Using a virtual environment (`.venv`) is the best practice for Python projects. It isolates your project's dependencies from your system-wide Python installation, preventing conflicts.

Here is the step-by-step process for setting up your environment and installing dependencies:

### Data Source Requirement

⚠️ **IMPORTANT**:

- You must provide your own custom JSON files (`wix_posts_data.json`, `wix_articles_data.json`, etc.) in the project's `/data` folder.
- You must also adapt the **`jq_schema`** variables in the code to match the structure of your custom data.

### Configuration (`.env`)

1. In the root directory of this project, rename the `.env.example` to `.env`.
2. Populate the file with your OpenRouter API key:

```dotenv
OPENROUTER_API_KEY=sk-or-v1-Your_OpenRouter_API_Key
```

⚠️ **Security Tip**: Never commit your `.env` file to version control.

### Setup Python Virtual Environment

It is best practice to use a virtual environment to isolate project dependencies.

- #### 1. Create the environment

Run the following command in your project directory:

```Bash
python3 -m venv .venv
```

- #### 2. Activate the environment

macOS/Linux:

```Bash
source .venv/bin/activate
```

Windows (Command Prompt):

```Bash
.venv\Scripts\activate.bat
```

Windows (PowerShell):

```Bash
.venv\Scripts\Activate.ps1
```

Your command prompt will now show the environment name, like `(.venv) user@host:~/project$`, indicating that it is active.

### Update pip

```Bash
python -m pip install --upgrade pip
```

### Install Dependencies

With the virtual environment active, install all necessary packages in a single command.

```bash
 pip install -U \
  langchain \
  langchain-community \
  langchain-openai \
  langchain-huggingface \
  langchain-chroma \
  chromadb \
  sentence-transformers \
  langchain-text-splitters \
  python-dotenv \
  jq \
  streamlit  # Required if you plan to run ContentNavigatorAI
```

### Run the Application

```Bash
# InsightHubAI (CLI)
python3 insight_hub_ai.py

# ContentNavigatorAI (Streamlit)
streamlit run content_navigator_ai.py
```

### Shut Down and Deactivate

When you are finished with development or testing, follow these steps to properly shut down all components and exit the environment.

1. Stop the Servers  
   If you are running **ContentNavigatorAI** (Streamlit), go to the terminal window running the server and press `Ctrl + C` (Control + C) to shut down the server process.

2. Deactivate the Environment  
   In any active terminal where the virtual environment is running, run the command:

```Bash
deactivate
```

Your command prompt will return to its default state, and the environment name (`.venv`) will disappear.
