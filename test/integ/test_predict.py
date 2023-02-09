from unittest.mock import MagicMock

import pandas as pd
import pytest
from assertpy import assert_that
from fastapi.testclient import TestClient

from src.dependencies import get_books_df, get_book_id_to_f_book_id, get_user_id_to_f_user_id, Properties
from src.main import app
from src.service.user_info_client import UserInfoClient, get_user_info_client, BooksReadResponse, \
    UserInfoServerException, UserInfoClientException


@pytest.fixture()
def user_info_client_mock():
    user_info_client = UserInfoClient(Properties())
    user_info_client.get_books_read = MagicMock(return_value=BooksReadResponse(book_ids=[4, 5, 6]))
    yield user_info_client


@pytest.fixture(autouse=True)
def run_around_tests(user_info_client_mock):
    # Code run before all tests
    _stub_dataframe_dependency()
    _stub_book_id_to_f_book_id()
    _stub_user_id_to_f_user_id()
    _stub_user_info_client(user_info_client_mock)

    yield
    # Code that will run after each test
    app.dependency_overrides = {}


def test_router_works(test_client: TestClient):
    response = test_client.get("/predict/1")
    assert_that(response.status_code).is_equal_to(200)


@pytest.mark.parametrize("count, expected_code", [(-1, 422), (50, 200), (101, 422)])
def test_query_validation_count_between_0_and_100(count, expected_code, test_client: TestClient):
    response = test_client.get("/predict/1?count={}".format(count))
    assert_that(response.status_code).is_equal_to(expected_code)


def test_count_works_to_limit_results(test_client: TestClient):
    response = test_client.get("/predict/1?count=1")
    assert_that(response.json().get("items")).is_length(1)


@pytest.mark.parametrize("genre_list, expected_count", [([], 3),
                                                        (["young_adult"], 1),
                                                        (["young_adult", "romance"], 1),
                                                        (["young_adult", "science"], 0)])
def test_genre_filters_correctly_filter_down_results(genre_list, expected_count, test_client: TestClient):
    genre_query_params = "&".join(["genres={}".format(genre) for genre in genre_list])
    response = test_client.get("/predict/1?{}".format(genre_query_params))
    assert_that(response.json().get("items")).is_length(expected_count)


@pytest.mark.parametrize("books_read, expected_count", [([1, 2, 3], 0),
                                                        ([1, 2], 1),
                                                        ([1], 2),
                                                        ([], 3)])
def test_previous_books_read_correctly_removes_book_from_suggestions(books_read,
                                                                     expected_count,
                                                                     user_info_client_mock: UserInfoClient,
                                                                     test_client: TestClient):
    # Given
    user_info_client_mock.get_books_read = MagicMock(return_value=BooksReadResponse(book_ids=books_read))

    # When
    response = test_client.get("/predict/1?")

    # Then
    assert_that(response.json().get("items")).is_length(expected_count)


def test_user_info_service_throwing_server_exception_doesnt_block_prediction(
        user_info_client_mock: UserInfoClient,
        test_client: TestClient):
    # Given
    user_info_client_mock.get_books_read = MagicMock(side_effect=UserInfoServerException("Boom"))

    # When
    response = test_client.get("/predict/1?")

    # Then
    assert_that(response.json().get("items")).is_length(3)


def test_user_info_service_throwing_client_exception_doesnt_block_prediction(
        user_info_client_mock: UserInfoClient,
        test_client: TestClient):
    # Given
    user_info_client_mock.get_books_read = MagicMock(side_effect=UserInfoClientException("Boom"))

    # When
    response = test_client.get("/predict/1?")

    # Then
    assert_that(response.json().get("items")).is_length(3)


def _stub_dataframe_dependency():
    input_books = [[1, "The Proposal", 3.49, 103443, 325.0, 52240, 59474,
                    "https://www.goodreads.com/author/show/16287225.Jasmine_Guillory", 1,
                    "/book/show/37584991-the-proposal", 399587683.0, 9780399587689.0, "B0782YRL2G",
                    "English", 16287225, "/book/show/37584991-the-proposal", False, False, False, True,
                    False, False, False, False, False, False, False, False, False, False, True, False,
                    True, False, False, False, False, False, False, False, False, False, False, False,
                    False, False, False, False, False, False, False, False, False, False, True, False,
                    0.0240306650486352, -1.4242729718515512, 0.1938119096081405, 1.0740150533532749],
                   [2, "The Chain", 3.8, 85179, 357.0, 55575, 33810,
                    "https://www.goodreads.com/author/show/12433.Adrian_McKinty", 2,
                    "/book/show/42779092-the-chain", None, 9780316531269, "031653126X", "English", 12433,
                    "/book/show/42779092-the-chain", False, False, False, False, False, False, False, False, False,
                    False, True, True, False, False, False, True, False, False, False, False, False, False, False,
                    False, False, False, True, False, False, False, False, True, False, False, False, False, False,
                    False, True, False, 0.0418499896502789, -0.4579905871559674, 0.2142191968111878, 0.547506711217822],
                   [3, "2 States: The Story of My Marriage", 3.43, 92423, 269.0, 44978, 55063,
                    "https://www.goodreads.com/author/show/61124.Chetan_Bhagat", 3,
                    "/book/show/6969361-2-states", 8129115301.0, 9788129115300.0, 8129115301, "English", 61124,
                    "/book/show/6969361-2-states", False, False, True, False, False, False, False, False, False, False,
                    False, False, False, False, True, False, True, False, False, False, False, False, False, False,
                    False, False, False, False, False, False, False, False, False, False, False, False, False, False,
                    True, False, -0.0071531530042412, -1.6112953688894067, 0.1493748122562578, 0.983521432048744]]

    dataframe = pd.DataFrame(input_books,
                             columns=["0", "book_title", "avg_rating", "num_ratings", "num_pages", "promoters",
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
                                      "scaled_avg_rating", "scaled_promoters", "scaled_detractors"])

    app.dependency_overrides[get_books_df] = lambda: dataframe


def _stub_book_id_to_f_book_id():
    app.dependency_overrides[get_book_id_to_f_book_id] = lambda: {1: 1, 2: 2, 3: 3}


def _stub_user_id_to_f_user_id():
    app.dependency_overrides[get_user_id_to_f_user_id] = lambda: {1: 1}


def _stub_user_info_client(user_info_client_mock):
    app.dependency_overrides[get_user_info_client] = lambda: user_info_client_mock
