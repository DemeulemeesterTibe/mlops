import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import os

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

ANIMALS = ['Cat', 'Dog', 'Panda']
# model_path = os.path.join("animal-cnn")
# model_path = os.path.join("inference/animals-classification/INPUT_model_path/animal-cnn")
model_path = os.path.join("inference", "animals-classification", "INPUT_model_path", "animal-cnn")
model = load_model(model_path)

@app.post('/upload/image')
async def uploadImage(img: UploadFile = File(...)):
    original_image = Image.open(img.file)
    original_image = original_image.resize((64, 64))
    images_to_predict = np.expand_dims(np.array(original_image), axis=0)
    predictions = model.predict(images_to_predict)
    classifications = predictions.argmax(axis=1)

    return ANIMALS[classifications.tolist()[0]]

@app.get('/healthcheck')
def healthcheck():
    return {'status': 'Healty'}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run("main:app", host='0.0.0.0', port=8000, reload=True)