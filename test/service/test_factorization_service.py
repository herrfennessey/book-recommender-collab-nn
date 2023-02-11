from assertpy import assert_that

from src.service.factorization_service import FactorizationService


def test_factorize_user_id_returns_when_exists():
    # Given
    factorization_service = FactorizationService(user_to_factorized={1: 2}, book_to_factorized={1: 2})
    user_id = 1
    expected_factorized_user_id = 2

    # When
    factorized_user_id = factorization_service.factorize_user_id(user_id)

    # Then
    assert_that(factorized_user_id).is_equal_to(expected_factorized_user_id)


def test_factorize_user_id_returns_none_when_not_exists():
    # Given
    factorization_service = FactorizationService(user_to_factorized={1: 2}, book_to_factorized={1: 2})
    user_id = 3
    expected_factorized_user_id = None

    # When
    factorized_user_id = factorization_service.factorize_user_id(user_id)

    # Then
    assert_that(factorized_user_id).is_equal_to(expected_factorized_user_id)


def test_factorize_book_id_returns_when_exists():
    # Given
    factorization_service = FactorizationService(user_to_factorized={1: 2}, book_to_factorized={1: 2})
    book_id = 1
    expected_factorized_book_id = 2

    # When
    factorized_book_id = factorization_service.factorize_book_id(book_id)

    # Then
    assert_that(factorized_book_id).is_equal_to(expected_factorized_book_id)


def test_factorize_book_id_returns_none_when_not_exists():
    # Given
    factorization_service = FactorizationService(user_to_factorized={1: 2}, book_to_factorized={1: 2})
    book_id = 3
    expected_factorized_book_id = None

    # When
    factorized_book_id = factorization_service.factorize_book_id(book_id)

    # Then
    assert_that(factorized_book_id).is_equal_to(expected_factorized_book_id)
