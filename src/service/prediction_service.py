import logging
import time
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
from fastapi import Depends
from pydantic import BaseSettings

from src.dependencies import get_model, get_books_df
from src.ml.ncf import NCF
from src.models.book_size import BookSize
from src.models.genre_list import GenreList
from src.service.factorization_service import FactorizationService, get_factorization_service
from src.service.user_info_client import UserInfoClient, get_user_info_client, UserInfoClientException, \
    UserInfoServerException

logger = logging.getLogger(__name__)

MAX_RECOMMENDATION_COUNT = 100


class PredictionServiceItem(BaseSettings):
    book_id: int
    book_title: Optional[str] = None
    author: Optional[str] = None
    score: float


class PredictionServiceResponse(BaseSettings):
    items: List[PredictionServiceItem]
    count: int = 20
    took_ms: int


class PredictionService:
    def __init__(self, model: NCF, books_dataframe: pd.DataFrame, user_info_client: UserInfoClient,
                 factorization_service: FactorizationService):
        self.model = model
        self.books_dataframe = books_dataframe
        self.user_info_client = user_info_client
        self.factorization_service = factorization_service

    def predict(self, user_id, genres: List[GenreList], book_size: BookSize,
                count: int = 20) -> PredictionServiceResponse:
        start_time = time.time()
        logger.info("Getting %d book predictions for user %s with genres: %s, for book_size: %s", count,
                    user_id, genres, book_size)

        books_read = self._get_books_read(user_id)
        candidate_df = self._get_candidates(genres, books_read)
        scored_candidates = self._score_candidates_for_user(candidate_df, user_id)

        candidate_items = [PredictionServiceItem(**item) for item in scored_candidates][:count]
        took_ms = (time.time() - start_time) * 1000
        return PredictionServiceResponse(items=candidate_items, count=len(scored_candidates), took_ms=took_ms)

    def _get_books_read(self, user_id):
        try:
            books_read = self.user_info_client.get_books_read(user_id)
            logger.info("User %s has read %d books", user_id, len(books_read.book_ids))
            return books_read.book_ids
        except (UserInfoClientException, UserInfoServerException):
            return []

    def _get_candidates(self, genres: List[GenreList], books_read: List[int]):
        all_candidates = self.books_dataframe.copy()
        if len(genres) > 0:
            query = " & ".join(f"{genre.name} == 1" for genre in genres)
            all_candidates.query(query, inplace=True)

        # Remove books already read
        all_candidates = all_candidates[~all_candidates['book_id'].isin(books_read)]
        return all_candidates

    def _score_candidates_for_user(self, candidate_df: pd.DataFrame, user_id: int):
        factorized_user_id = self.factorization_service.factorize_user_id(user_id)
        candidate_df.insert(0, 'f_user_id', factorized_user_id)
        candidate_df['f_book_id'] = candidate_df['book_id'].map(self.factorization_service.factorize_book_id)

        predicted_labels = np.squeeze(self.model(
            torch.tensor(candidate_df.loc[:, "f_user_id"].values),
            torch.tensor(candidate_df.loc[:, "f_book_id"].values),
            torch.FloatTensor(candidate_df.loc[:, [col for col in candidate_df if col.startswith('scaled')]].values),
            torch.BoolTensor(
                candidate_df.loc[:, [col for col in candidate_df if col.startswith('genre')]].values.astype(bool))
        ).detach().numpy())

        candidate_df['score'] = predicted_labels
        # Returns the top N books with the highest score into a list of dictionaries
        return candidate_df.nlargest(MAX_RECOMMENDATION_COUNT, 'score')[['book_id', 'book_title', 'score']].to_dict(
            orient='records')


def get_prediction_service(model: NCF = Depends(get_model),
                           books_df: pd.DataFrame = Depends(get_books_df),
                           user_info_client: UserInfoClient = Depends(get_user_info_client),
                           factorization_service: FactorizationService = Depends(get_factorization_service)
                           ) -> PredictionService:
    """
    Used for FastAPI dependency injection
    """
    return PredictionService(model=model, books_dataframe=books_df, user_info_client=user_info_client,
                             factorization_service=factorization_service)
