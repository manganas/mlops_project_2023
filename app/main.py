import torch
import torch.nn.functional as F
from PIL import Image
from pathlib import Path
from typing import List, Optional
from http import HTTPStatus
import numpy as np
from transformers import AutoFeatureExtractor

import pickle

from fastapi import FastAPI
from pydantic import BaseModel
from fastapi import UploadFile, File

app = FastAPI(
    title="Team 31 - Image Classifier",
    description="Classify bird photos to their commona name species.",
    version="0.1",
)


saved_path = "extra_files"

for file_ in Path(saved_path).glob("**/*checkpoint*.pt"):
    model_path = file_.as_posix()

for file_ in Path(saved_path).glob("**/feature_extractor.pt"):
    extractor_path = file_.as_posix()

model = torch.load(model_path)
# feature_extractor = torch.load(extractor_path)

pretrained = 'google/vit-base-patch16-224-in21k'
feature_extractor = AutoFeatureExtractor.from_pretrained(pretrained, cache_dir='./feat_extr')

device = torch.device( 'cuda' if torch.cuda.is_available() else 'cpu' )

with open(saved_path+'/id2label.pkl', 'rb') as f:
    id2label = pickle.load(f)



model.to(device)

@app.get("")
@app.get('/')
def root():
    response = {
        'message': HTTPStatus.OK.phrase,
        'status-code': HTTPStatus.OK
    }
    return response

@app.post('/predict')
async def predict(data: UploadFile = File(...)):
    with open('image.png', 'wb') as image:
        content = await data.read()
        image.write(content)
        image.close()


    image = Image.open('image.png')
    if image.mode != "RGB":
         image = image.convert(mode="RGB")

    pixel_values = feature_extractor(image, return_tensors="pt").pixel_values

    pixel_values = pixel_values.to(device)

    with torch.no_grad():
        pred_logits =  model(pixel_values=pixel_values).logits
        pred_probs = F.softmax(pred_logits, dim=1)
        pred_class = torch.argmax(pred_probs, dim=1).cpu().numpy()
        p = pred_class[0]


    response = {
        'message': HTTPStatus.OK.phrase,
        'status-code': HTTPStatus.OK,
        'prediction': f"{id2label[p]}",
        'probability': f"{pred_probs[0,p]}"
    }

    return response
