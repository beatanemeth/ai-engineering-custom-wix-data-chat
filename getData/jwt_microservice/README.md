# FastAPI/JWT Microservice

> In case, that you only would like to use this service, follow the steps bellow for the installation.

## Getting Started 🚀

Using a virtual environment (`.venv`) is the best practice for Python projects. It isolates your project's dependencies from your system-wide Python installation, preventing conflicts.

Here is the step-by-step process for setting up your environment and installing dependencies:

### Configuration (`.env`)

1. In the root directory of this project, rename the `.env.example` to `.env`.
2. Populate the file with your environmental keys:

   - `WIX_AUTH_SECRET`: The secret key used to sign the JWT (must match the secret on the Wix side).
   - `JWT_SUBJECT_*` values (e.g., `JWT_SUBJECT_EVENTS`): The subject (`sub` claim) for each service.
   - `WIX_*_ENDPOINT` values (e.g., `WIX_EVENTS_ENDPOINT`): The full Wix HTTP function URL.

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

```Bash
pip install fastapi uvicorn requests pyjwt python-dotenv
```

### Run the Application

#### Terminal_1: Run the Service

Change Directory: Navigate to the service directory where `main.py` is located (e.g., `/getData/jwt_microservice/`).

```Bash
cd getData/jwt_microservice/
```

Start the Application: Since you installed `python-dotenv`, the server will automatically load the secret key and endpoint URLs from your local `.env` file.

```Bash
# The application loads WIX_AUTH_SECRET and other variables from the .env file.
uvicorn main:app --reload
```

Leave this terminal open.

---

#### Terminal_2: Call the Endpoint:

Open a second terminal window and perform the following steps:

1. **Activate the Environment**: You must activate the environment here as well to access the project utilities, though curl is system-level.

```Bash
source .venv/bin/activate
```

2. **Execute the Request**: Execute the `curl` command to call an endpoint. The service runs on `http://127.0.0.1:8000` by default.

```Bash
curl http://127.0.0.1:8000/downloadEvents

# OR

curl http://127.0.0.1:8000/downloadPosts

# OR

curl http://127.0.0.1:8000/downloadArticles
```

You will see the output (the JSON response) in this second terminal window.

Simultaneously, in your first terminal (where `Uvicorn` is running), you will see log messages confirming the request was received and processed, such as:

```bash
INFO: 127.0.0.1:XXXXX - "GET /downloadEvents HTTP/1.1" 200 OK
```

If successful, the downloaded file (e.g., `wix_events_data.json`) will be saved to your project's `/data `directory (relative to your project root).

### Deactivate the Environment

#### Terminal 1: Stop the Server

Stop the server by pressing `Ctrl + C` (Control + C) on your keyboard. This shuts down the Uvicorn server.

#### Both Terminals: Exit the Environment

When you are finished, exit the isolated environment in both terminals:

```Bash
deactivate
```

Your command prompt will return to its default state, and the environment name (`.venv`) will disappear.
