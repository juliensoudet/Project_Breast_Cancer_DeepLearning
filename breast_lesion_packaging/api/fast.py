from fastapi import FastAPI, UploadFile, File, Response
from fastapi.responses import FileResponse
import requests
from fastapi.middleware.cors import CORSMiddleware
from google.cloud import storage
from breast_lesion_DL_pack.params import *
import numpy as np
from PIL import Image
import cv2
import os
from tensorflow.keras.models import load_model
from breast_lesion_DL_pack.predictor import *
import tensorflow as tf

app = FastAPI()

# Allowing all middleware is optional, but good practice for dev purposes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

full_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
app.state.model = load_model(os.path.join(full_path,'models','model_best.keras'))

@app.get("/")
def root():
    return dict(greeting="Welcome to the breast lesion project ")


@app.post("/predict")
async def receive_image(img: UploadFile=File(...)):

    try:
    ### Receiving and decoding the image
        contents = await img.read()

        nparr = np.fromstring(contents, np.uint8)

        #print(resized_nparr.shape)
        cv2_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR) # type(cv2_img) => numpy.ndarray

        #image_opened = Image.fromarray(cv2_img,'RGB')
        #image_opened = image_opened.convert("L")
        resized_image = cv2.resize(cv2_img, (224,224))
        resized_image2 = np.expand_dims(resized_image,axis=0)



        result = app.state.model.predict(resized_image2)[0].tolist()
        answer_text = test(result)

        return answer_text



    except Exception as e:
            # If an exception occurs during the request, print the error message
        print(f"Error fetching image. Exception: {e}")
        return None
