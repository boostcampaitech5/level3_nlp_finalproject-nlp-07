from fastapi import FastAPI
from typing import List
from pydantic import BaseModel
from model import DenseRetriever
from starlette.middleware.cors import CORSMiddleware


class Item(BaseModel):
    query: str
    reviews: List[str]


class DBItem(BaseModel):
    query: str
    reviews: List[dict]


app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


DPR = DenseRetriever()


@app.post("/dpr/concat")
def retrieve_concat(item: Item):
    return {"review": DPR.run_dpr_concat(item.query, item.reviews)}


@app.post("/dpr/split")
def retrieve_split(item: Item):
    return {"review": DPR.run_dpr_split(item.query, item.reviews)}


@app.post("/dpr/split_db")
def retrieve_split_db(item: DBItem):
    return {"review": DPR.run_dpr_db(item.query, item.reviews)}


@app.post("/dpr/split_db_v3")
def retrieve_split_db_v3(item: DBItem):
    return {"review": DPR.run_dpr_db_v3(item.query, item.reviews)}
