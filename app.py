from fastapi.exceptions import RequestValidationError
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from fastapi import FastAPI
import json
import joblib
from pathlib import Path
import pandas as pd
import uvicorn

from src.schema import HouseProperty

model_path = Path.cwd() / "models" / "model.pkl"
model = joblib.load(model_path)

app = FastAPI()


@app.get("/")
def read_root():
    return {"response": "Ready!"}


@app.get("/model-info")
def get_model_info():
    model_info_path = Path.cwd() / "models" / "model_info.json"
    with open(model_info_path) as f:
        model_info = json.load(f)
    return model_info


@app.post("/predict")
def predict(property: HouseProperty):
    try:
        property = jsonable_encoder(property)
        property = pd.DataFrame([property])
        y_pred = model.predict(property)
        price = int(y_pred[0])
        return {"prediction": price, "status_code": status.HTTP_200_OK}
    except RequestValidationError as e:
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=e,
        )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    key = exc.errors()[0]["loc"][1]
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"error": f"{key} {exc.errors()[0]['msg']}"},
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
