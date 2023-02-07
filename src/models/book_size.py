from enum import Enum


class BookSize(str, Enum):
    book_size_small = "small"
    book_size_medium = "medium"
    book_size_large = "large"
