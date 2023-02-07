from __future__ import annotations

import logging.config
import uuid
from os import path

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette import status

from src.dependencies import initialize_dependencies, get_model_properties, validate_dependencies
from src.routers import predict

# setup loggers to display more information
log_file_path = path.join(path.dirname(path.abspath(__file__)), "logging.conf")
logging.config.fileConfig(log_file_path, disable_existing_loggers=False)

# get root logger
logger = logging.getLogger(__name__)

app = FastAPI()


# Process the dependencies once at startup so we don't incur costs on each call
@app.on_event("startup")
def startup():
    initialize_dependencies()
    validate_dependencies()


@app.get("/", tags=["welcome"])
def welcome():
    return {"status": "Ready to Rock!"}


@app.get("/info")
def model_info():
    return get_model_properties()


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    uuid_str = str(uuid.uuid4())
    exc_str = f"{uuid_str} - {exc}".replace("\n", " ").replace("   ", " ")
    logger.error(exc_str)
    content = {"status_code": 10422, "message": exc_str, "data": None}
    return JSONResponse(
        content=content, status_code=status.HTTP_422_UNPROCESSABLE_ENTITY
    )


app.include_router(predict.router)
