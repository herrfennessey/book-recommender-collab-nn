import httpx
import pytest
from assertpy import assert_that

from src.dependencies import Properties
from src.service.user_info_client import BooksReadResponse, UserInfoClient, UserInfoClientException, \
    UserInfoServerException

TEST_PROPERTIES = Properties(book_recommender_api_base_url="https://testurl", env_name="test")


def test_successful_response_from_user_info_client(httpx_mock):
    httpx_mock.add_response(json={'book_ids': [1, 2, 3]}, url="https://testurl/users/1/books-read")

    user_id = 1
    client = UserInfoClient(properties=TEST_PROPERTIES)
    response = client.get_books_read(user_id)
    assert response == BooksReadResponse(book_ids=[1, 2, 3])


@pytest.mark.parametrize("response_code", [500, 501, 502, 503, 504])
def test_5xx_custom_exception_from_user_info_client(response_code, httpx_mock):
    httpx_mock.add_response(status_code=response_code, url="https://testurl/users/1/books-read")

    user_id = 1
    client = UserInfoClient(properties=TEST_PROPERTIES)
    assert_that(client.get_books_read).raises(UserInfoServerException).when_called_with(user_id)


@pytest.mark.parametrize("response_code", [400, 401, 402, 403, 404])
def test_4xx_custom_exception_from_user_info_client(response_code, httpx_mock):
    httpx_mock.add_response(status_code=response_code, url="https://testurl/users/1/books-read")

    user_id = 1
    client = UserInfoClient(properties=TEST_PROPERTIES)
    assert_that(client.get_books_read).raises(UserInfoClientException).when_called_with(user_id)


def test_uncaught_exception_from_user_info_client(httpx_mock):
    httpx_mock.add_exception(httpx.ReadTimeout("Unable to read within timeout"))

    user_id = 1
    client = UserInfoClient(properties=TEST_PROPERTIES)
    assert_that(client.get_books_read).raises(UserInfoServerException).when_called_with(user_id)
