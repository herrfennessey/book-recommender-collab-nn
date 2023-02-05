import pandas as pd

from src.model.ncf import NCF


class PredictionService:
    def __init__(self, model: NCF, books_dataframe: pd.DataFrame):
        self.model = model
        self.books_dataframe = books_dataframe

    def predict(self, data):
        return self.model.predict(data)
