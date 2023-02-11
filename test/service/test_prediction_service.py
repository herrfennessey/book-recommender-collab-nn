from unittest.mock import patch

import pandas as pd
import pytest
from assertpy import assert_that

from src.dependencies import get_model
from src.ml.ncf import NCF
from src.service.factorization_service import FactorizationService
from src.service.prediction_service import PredictionService, UserNotFoundException
from src.service.user_info_client import UserInfoClient, BooksReadResponse


@pytest.fixture()
def user_info_client():
    with patch('src.service.user_info_client.UserInfoClient') as mock_user_info_client:
        mock_user_info_client.get_books_read.return_value = BooksReadResponse(book_ids=[])
        yield mock_user_info_client


@pytest.fixture()
def factorization_service():
    with patch('src.service.factorization_service.FactorizationService') as mock_factorization_service:
        # Map user ID 1:1 to factorized ID
        mock_factorization_service.factorize_user_id = lambda user_id: user_id
        # Map book ID 1:1 to factorized ID
        mock_factorization_service.factorize_book_id = lambda book_id: book_id
        yield mock_factorization_service


@pytest.fixture()
def model():
    yield get_model()


def test_result_set_truncated_to_100_regardless_of_count(model: NCF,
                                                         user_info_client: UserInfoClient,
                                                         factorization_service: FactorizationService):
    # Given
    df_data = [_generate_dummy_book(idx) for idx in range(0, 200)]
    dataframe = pd.DataFrame(df_data, columns=_get_df_columns())
    pred_service = PredictionService(model, dataframe, user_info_client, factorization_service)

    # When
    result = pred_service.predict(1, [], count=200)

    # Then
    assert_that(result.items).is_length(100)
    assert_that(result.count).is_equal_to(100)


def test_missing_factorized_user_id_fails_with_exception(model: NCF,
                                                         user_info_client: UserInfoClient,
                                                         factorization_service: FactorizationService):
    # Given
    dataframe = pd.DataFrame([_generate_dummy_book(1)], columns=_get_df_columns())
    # Simulate user not found
    factorization_service.factorize_user_id = lambda user_id: None
    pred_service = PredictionService(model, dataframe, user_info_client, factorization_service)

    # When / Then
    assert_that(pred_service.predict).raises(UserNotFoundException).when_called_with(1)


def test_missing_factorized_book_id_drops_it_from_recommendations(model: NCF,
                                                                  user_info_client: UserInfoClient,
                                                                  factorization_service: FactorizationService):
    # Given
    dataframe = pd.DataFrame([_generate_dummy_book(idx) for idx in range(1, 4)], columns=_get_df_columns())
    # Simulate book not found for second book
    factorization_service.factorize_book_id = lambda book_id: None if book_id == 2 else int(book_id)
    pred_service = PredictionService(model, dataframe, user_info_client, factorization_service)

    # When
    results = pred_service.predict(1)

    # Then
    assert_that(results.items).is_length(2)


def _get_df_columns():
    return ["0", "book_title", "avg_rating", "num_ratings", "num_pages", "promoters",
            "detractors", "author_url", "book_id", "book_url", "isbn", "isbn13", "asin",
            "language", "author_id", "book_url_1", "genre_science", "genre_biography",
            "genre_young_adult", "genre_chick_lit", "genre_psychology", "genre_ebooks",
            "genre_religion", "genre_cookbooks", "genre_humor_and_comedy", "genre_self_help",
            "genre_crime", "genre_mystery", "genre_manga", "genre_paranormal",
            "genre_romance", "genre_horror", "genre_contemporary", "genre_music",
            "genre_science_fiction", "genre_graphic_novels", "genre_art", "genre_history",
            "genre_memoir", "genre_childrens", "genre_gay_and_lesbian", "genre_poetry",
            "genre_thriller", "genre_historical_fiction", "genre_philosophy",
            "genre_spirituality", "genre_comics", "genre_suspense", "genre_fantasy",
            "genre_nonfiction", "genre_travel", "genre_business", "genre_classics",
            "genre_christian", "genre_fiction", "genre_sports", "scaled_num_pages",
            "scaled_avg_rating", "scaled_promoters", "scaled_detractors"]


def _generate_dummy_book(book_id):
    return [1, "The Proposal", 3.49, 103443, 325.0, 52240, 59474,
            "https://www.goodreads.com/author/show/16287225.Jasmine_Guillory", book_id,
            "/book/show/37584991-the-proposal", 399587683.0, 9780399587689.0, "B0782YRL2G",
            "English", 16287225, "/book/show/37584991-the-proposal", False, False, False, True,
            False, False, False, False, False, False, False, False, False, False, True, False,
            True, False, False, False, False, False, False, False, False, False, False, False,
            False, False, False, False, False, False, False, False, False, False, True, False,
            0.0240306650486352, -1.4242729718515512, 0.1938119096081405, 1.0740150533532749]
