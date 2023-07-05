from fastapi import FastAPI
from typing import List
from pydantic import BaseModel
from model import DenseRetriever


class Item(BaseModel):
    query: str
    reviews: List[str]


app = FastAPI()

DPR = DenseRetriever()


@app.get("/dpr/concat")
def retrieve_concat(item: Item):
    return {"review": DPR.run_dpr_concat(item.query, item.reviews)}


@app.get("/dpr/split")
def retrieve_split(item: Item):
    return {"review": DPR.run_dpr_split(item.query, item.reviews)}
