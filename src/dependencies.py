import logging
import os
import pickle
import time
from pathlib import Path

import pandas
import pandas as pd
import torch
from pydantic import BaseSettings

from src.ml.ncf import NCF


class Properties(BaseSettings):
    env_name: str = "local"
    book_recommender_api_base_url: str = "http://localhost:8999"


root_path = Path(os.getenv("MODEL_FOLDER", "."))

book_id_to_f_book_id = {}
f_book_id_to_book_id = {}
f_user_id_to_user_id = {}
user_to_books_read = {}
model_properties = {}
model = None
books_df = None


def initialize_dependencies():
    logging.warning("Initializing dependencies")
    start_time = time.time()
    global book_id_to_f_book_id
    global f_book_id_to_book_id
    global f_user_id_to_user_id
    global user_to_books_read
    global model
    global model_properties
    global books_df

    book_id_to_f_book_id = pickle.load(open(root_path / "book_id_to_f_book_id.p", "rb"))
    f_book_id_to_book_id = pickle.load(open(root_path / "f_book_id_to_book_id.p", "rb"))
    f_user_id_to_user_id = pickle.load(open(root_path / "f_user_id_to_user_id.p", "rb"))
    model_properties = pickle.load(open(root_path / "model_properties.p", "rb"))

    books_df = pandas.read_csv(root_path / "books.csv")

    # Stand up model and load weights
    model_weights = torch.load(root_path / "model_weights.pth")
    model = NCF(pd.DataFrame, model_properties.get("num_users"), model_properties.get("num_books"))
    model.load_state_dict(model_weights)
    model.eval()

    logging.warning("Dependencies initialized in %s seconds", time.time() - start_time)


def validate_dependencies():
    assert len(get_book_id_to_f_book_id()) > 0, "book_id_to_f_book_id not initialized"
    assert len(get_f_book_id_to_book_id()) > 0, "f_book_id_to_book_id not initialized"
    assert len(get_f_user_id_to_user_id()) > 0, "f_user_id_to_user_id not initialized"
    assert len(get_model_properties()) > 0, "model_properties not initialized"
    assert type(get_model()) == NCF, "ml not initialized"
    assert get_books_df() is not None, "books_df not initialized"
    logging.warning("Dependencies validated! Ready to Rock!")


def get_book_id_to_f_book_id() -> dict:
    return book_id_to_f_book_id


def get_f_book_id_to_book_id() -> dict:
    return f_book_id_to_book_id


def get_f_user_id_to_user_id() -> dict:
    return f_user_id_to_user_id


def get_model_properties() -> dict:
    return model_properties


def get_model() -> NCF:
    return model


def get_books_df() -> pd.DataFrame:
    return books_df
