from fastapi import FastAPI, UploadFile, File
from keras.models import load_model
import cv2
import numpy as np

app = FastAPI()

model_path = 'animals_classification'
model = load_model(model_path)

def preprocess_image(image):
    resized_image = cv2.resize(image, (64, 64))
    preprocessed_image = np.expand_dims(resized_image, axis=0)
    return preprocessed_image

@app.post("/predict/")
async def predict(file: UploadFile):
    # Read and preprocess the uploaded image
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    preprocessed_image = preprocess_image(image)

    # Make a prediction using the Keras model
    prediction = model.predict(preprocessed_image)
    predicted_class = np.argmax(prediction[0])

    return {"predicted_class": predicted_class}

if __name__ == "__main__":
    import uvicorn

    # Run the FastAPI app using Uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
