# Book Recommender NN Collaborative Filtering API

This is a FastAPI application designed to serve recommendations from a PyTorch based collaborative filtering model. This
model requires both the training weights as well as the book dataset it is referencing in order to function correctly.

## API Documentation

The API documentation can be found at the root of the application. For example, if the application is running on
`localhost:9000`, the documentation can be found at `localhost:9000/docs`.

## API Endpoints

- `/predict/{user_id}`: Returns a list of recommended books for the given user ID. For more information, see the API
  documentation.

## Prerequisites

- Python 3.10+
- Collab NN model weights
- Book dataset
- [Optional] Docker
- [Optional] Docker Compose

## Run Instructions

1. (Optional) Run Wiremock to mock the Book API:

    ```
    git clone https://github.com/herrfennessey/book-recommender-wiremock/
    cd book-recommender-wiremock
    docker-compose up -d
    ```

2. Install the required dependencies:

    ```
    pip install -r requirements.txt
    ```

3. Run the application:

    ```
    cd book-recommender-collab-nn
    uvicorn main:app --reload --workers=1
    ```


