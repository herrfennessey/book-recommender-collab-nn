from fastapi import FastAPI

from src.dependencies import initialize_dependencies, get_model_properties, validate_dependencies

app = FastAPI()


# Process the dependencies once at startup so we don't incur costs on each call
@app.on_event("startup")
def startup():
    initialize_dependencies()
    validate_dependencies()


@app.get("/info")
def model_info():
    return get_model_properties()
