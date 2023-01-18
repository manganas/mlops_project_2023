from http import HTTPStatus

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(
    title="Team 31 - Image Classifier",
    description="Classify bird photos to their commona name species.",
    version="0.1",
)


@app.get("/")
def read_root():
    response = {"message": HTTPStatus.OK.phrase, "status-code": HTTPStatus.OK}
    return response


@app.get("/items/{item_id}")
def read_item(item_id: int):
    return {"item_id": item_id}


@app.get("/query_items")
def query_item(item_id: int):
    return {"item_id": item_id}


database = {
    "username": [],
    "password": [],
}


@app.post("/login/")
def login(username: str, password: str):
    usernames_db = database["username"]
    passwords_db = database["password"]
    if username not in usernames_db and password not in passwords_db:
        usernames_db.append(username)
        passwords_db.append(password)
        with open("database.csv", "a") as f:
            f.write(f"{username}, {password}\n")

    return "login_saved"


class Item(BaseModel):
    email: str
    domain_match: str


import re


@app.post("/text_model/")
def contains_email(data: Item):
    regex = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
    response = {
        "input": data.email,
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "is_email": re.fullmatch(regex, data.email) is not None,
    }
    return response


from fastapi import UploadFile, File
from typing import Optional
import cv2
from fastapi.responses import FileResponse


@app.post("/cv_model/")
async def cv_model(data: UploadFile = File(...)):
    with open("image.jpg", "wb") as f:
        content = await data.read()
        f.write(content)
        f.close()

    img = cv2.imread("image.jpg")
    img_res = cv2.resize(img, (50, 50))
    cv2.imwrite("image_resized.jpeg", img_res)

    response = {
        "input": data,
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "image": FileResponse("image_resized.jpeg"),
    }

    return response
