FROM python:3.10-slim
WORKDIR /code

COPY ./requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

ARG MODEL_FILES
COPY ./${MODEL_FILES}/ /code/model
COPY src /code/src

ENV MODEL_FOLDER=/code/model
CMD uvicorn src.main:app --host 0.0.0.0 --port $PORT --log-config /code/src/logging.conf --workers 1
