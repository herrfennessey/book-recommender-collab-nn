import logging
from functools import lru_cache
from typing import List

import httpx
from fastapi import Depends
from pydantic import BaseSettings, Extra

from src.dependencies import Properties


@lru_cache()
def get_properties():
    return Properties()


class UserInfoClient(object):
    """
    Client wrappper around the Book Recommender API. This API will be used to determine
    the books that a user has read. This information will be used to exclude books that
    the user has already read from the recommendations.
    """

    def __init__(self, properties):
        self.base_url = properties.book_recommender_api_base_url

    def get_books_read(self, user_id):
        url = self.base_url + "/users/" + str(user_id) + "/books-read"
        try:
            response = httpx.get(url)
            if not response.is_error:
                return BooksReadResponse(**response.json())
            elif response.is_client_error:
                logging.warning(
                    "{} status code encountered when querying {} "
                    "for user_id: {}".format(response.status_code, url, user_id)
                )
                raise UserInfoClientException("4xx Exception encountered for user_id: {}".format(user_id))
            elif response.is_server_error:
                logging.error(
                    "{} status code encountered when querying {} "
                    "for user_id: {}".format(response.status_code, url, user_id)
                )
                raise UserInfoServerException("5xx Exception encountered for user_id: {}".format(user_id))
        except httpx.HTTPError as e:
            logging.error("Uncaught Exception:{} encountered when querying {} for user_id: {}".format(e, url, user_id))
            raise UserInfoServerException("Uncaught Exception encountered for user_id: {}".format(user_id))


class BooksReadResponse(BaseSettings):
    book_ids: List[int]

    class Config:
        extra = Extra.ignore


class UserInfoClientException(Exception):
    pass


class UserInfoServerException(Exception):
    pass


def get_user_info_client(properties: Properties = Depends(get_properties)) -> UserInfoClient:
    """
    Used for FastAPI dependency injection
    """
    return UserInfoClient(properties=properties)
