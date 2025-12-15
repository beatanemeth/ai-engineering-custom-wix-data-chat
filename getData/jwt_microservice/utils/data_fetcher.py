import json
import requests
import os
from fastapi import HTTPException
from pathlib import Path
from utils.jwt_generator import generate_jwt


# Configuration
JWT_SECRET = os.environ.get("WIX_AUTH_SECRET")
DATA_DIR = Path("../../data")


async def data_fetcher(
    jwt_subject: str,
    wix_endpoint: str,
    local_path: str,
    data_type: str,
):
    """
    Calls the given Wix HTTP function, including a JWT token for authentication,
    and saves the resulting JSON data locally."""

    if not JWT_SECRET:
        raise HTTPException(
            status_code=500, detail="Server configuration error: JWT Secret is not set."
        )

    try:
        # Generate the JWT Token
        token = generate_jwt(secret=JWT_SECRET, subject=jwt_subject)

        # Define headers with the Authorization token
        headers = {"Authorization": f"Bearer {token}"}

        # Call the Wix HTTP Function with the token in headers
        response = requests.get(wix_endpoint, headers=headers, timeout=10)
        response.raise_for_status()
        fetched_data = response.json()

        # Ensure the output directory exists
        DATA_DIR.mkdir(parents=True, exist_ok=True)

        # Save the JSON data locally
        full_local_path = DATA_DIR / local_path
        with open(full_local_path, "w", encoding="utf-8") as f:
            json.dump(fetched_data, f, ensure_ascii=False, indent=4)

        return {
            "message": f"{data_type} data successfully retrieved and saved.",
            "file_path": str(full_local_path),
            "events_count": len(fetched_data),
        }

    # Catch all errors (Request errors, JSON errors, File errors)
    except Exception as e:
        error_msg = f"Could not retrieve {data_type} data or save file. Check the external service URL."  # Generalized message
        print(f"Internal error for logging ({data_type}): {e}")  # Generalized log

        raise HTTPException(status_code=503, detail=error_msg)
