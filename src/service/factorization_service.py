from typing import Dict

from fastapi import Depends

from src.dependencies import get_user_id_to_f_user_id, get_book_id_to_f_book_id


class FactorizationService:
    """
    Support class to factorize and defactorize user and book IDs. Factorization is a technique to lower the
    dimensionality and size of a model, and is used in the NCF model to shrink the size of the embedding matrix.

    These dictionaries are loaded at startup of this app, and are shipped along with the model definitions as a pickle.
    """

    def __init__(self,
                 user_to_factorized: Dict[int, int],
                 book_to_factorized: Dict[int, int],
                 ):
        self.user_to_factorize = user_to_factorized
        self.book_to_factorize = book_to_factorized

    def factorize_user_id(self, user_id):
        return self.user_to_factorize.get(user_id)

    def factorize_book_id(self, book_id):
        return self.book_to_factorize.get(book_id)


def get_factorization_service(
        user_to_factorized: Dict[int, int] = Depends(get_user_id_to_f_user_id),
        book_to_factorized: Dict[int, int] = Depends(get_book_id_to_f_book_id),
):
    return FactorizationService(user_to_factorized, book_to_factorized)
