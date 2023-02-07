import pytest
from assertpy import assert_that
from fastapi.testclient import TestClient


def test_router_works(test_client: TestClient):
    response = test_client.get("/predict/1?genres=fantasy&genres=romance&book_size=large")
    assert_that(response.status_code).is_equal_to(200)
    assert_that(response.json()).is_equal_to({"book_ids": [1, 2, 3]})


@pytest.mark.parametrize("count, expected_code", [(-1, 422), (50, 200), (101, 422)])
def test_query_validation_count_between_0_and_100(count, expected_code, test_client: TestClient):
    response = test_client.get("/predict/1?count={}".format(count))
    assert_that(response.status_code).is_equal_to(expected_code)
