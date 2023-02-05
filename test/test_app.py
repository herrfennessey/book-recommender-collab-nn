from assertpy import assert_that
from fastapi.testclient import TestClient


def test_read_main(test_client: TestClient):
    response = test_client.get("/")
    assert_that(response.status_code).is_equal_to(200)
    assert_that(response.json()).is_equal_to({"status": "Ready to Rock!"})


def test_reading_model_properties(test_client: TestClient):
    response = test_client.get("/info")
    assert_that(response.status_code).is_equal_to(200)
    assert_that(response.json()).contains_entry({"source_folder": "test_folder/2002-01-01"})
