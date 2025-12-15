# routes/events.py
import os
from fastapi import APIRouter
from utils import data_fetcher

# Initialize the APIRouter
router = APIRouter(
    tags=["Posts"],
)

# Configuration
JWT_SUBJECT = os.environ.get("JWT_SUBJECT_POSTS")
WIX_ENDPOINT = os.environ.get("WIX_POSTS_ENDPOINT")
LOCAL_PATH = "wix_posts_data.json"
DATA_TYPE = "Posts"


@router.get("/downloadPosts")
async def download_posts():
    """
    Calls the generalized data fetch utility for Posts data.
    """
    return await data_fetcher(
        jwt_subject=JWT_SUBJECT,
        wix_endpoint=WIX_ENDPOINT,
        local_path=LOCAL_PATH,
        data_type=DATA_TYPE,
    )
