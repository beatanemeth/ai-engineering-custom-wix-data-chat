# routes/events.py
import os
from fastapi import APIRouter
from utils import data_fetcher

# Initialize the APIRouter
router = APIRouter(
    tags=["Articles"],
)

# Configuration
JWT_SUBJECT = os.environ.get("JWT_SUBJECT_ARTICLES")
WIX_ENDPOINT = os.environ.get("WIX_ARTICLES_ENDPOINT")
LOCAL_PATH = "wix_articles_data.json"
DATA_TYPE = "Articles"


@router.get("/downloadArticles")
async def download_articles():
    """
    Calls the generalized data fetch utility for Articles data.
    """
    return await data_fetcher(
        jwt_subject=JWT_SUBJECT,
        wix_endpoint=WIX_ENDPOINT,
        local_path=LOCAL_PATH,
        data_type=DATA_TYPE,
    )
