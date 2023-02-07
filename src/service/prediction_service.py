import logging
from typing import List

import pandas as pd
from fastapi import Depends
from pydantic import BaseSettings

from src.dependencies import get_model, get_books_df
from src.ml.ncf import NCF
from src.models.book_size import BookSize
from src.models.genre_list import GenreList

logger = logging.getLogger(__name__)


class PredictionServiceResponse(BaseSettings):
    book_ids: List[int]


class PredictionService:
    def __init__(self, model: NCF, books_dataframe: pd.DataFrame):
        self.model = model
        self.books_dataframe = books_dataframe

    def predict(self, user_id, genres: List[GenreList], book_size: BookSize, num_books: int = 20):
        logger.info("Getting %d book predictions for user %s with genres: %s, for book_size: %s", num_books, user_id,
                    ", ".join(genres), book_size)
        return PredictionServiceResponse(book_ids=[1, 2, 3])


def get_prediction_service(model: NCF = Depends(get_model),
                           books_df: pd.DataFrame = Depends(get_books_df)) -> PredictionService:
    """
    Used for FastAPI dependency injection
    """
    return PredictionService(model=model, books_dataframe=books_df)
