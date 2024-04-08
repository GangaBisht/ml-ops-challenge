from fastapi import FastAPI
from api.predict import router as PredictRouter

app = FastAPI()

app.include_router(PredictRouter, tags=["predict"])

