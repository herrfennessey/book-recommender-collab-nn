import logging
from typing import List

from fastapi import APIRouter, Query, Path, Depends

from src.models.genre_list import GenreList
from src.service.prediction_service import PredictionService, get_prediction_service, PredictionServiceResponse

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/predict")


@router.get("/{user_id}", tags=["prediction"], status_code=200)
async def get_book_predictions(
        user_id: int = Path(
            title="The user ID from the Goodreads profile",
            gt=0,
            example=2189273),
        genres: List[GenreList] = Query(list()),
        count: int = Query(20, gt=0, le=100),
        prediction_service: PredictionService = Depends(get_prediction_service)) -> PredictionServiceResponse:
    """
    Get recommendations for a given user ID, if we've never seen the user before, it'll throw a 404.
    """
    return prediction_service.predict(user_id, genres, count)
