import logging

from fastapi import APIRouter

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/users")


@router.post("/predict", tags=["prediction"], status_code=200)
async def get_book_predictions():
    """
    Get book predictions for a user
    """
    pass
