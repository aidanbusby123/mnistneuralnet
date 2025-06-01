from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
from network import Network
from fastapi.middleware.cors import CORSMiddleware

# Initialize FastAPI app
app = FastAPI(root_path="/projects/MNISTneuralnet")

# Add CORS middleware
origins = [
    "https://busbylabs.com",  # Replace with your production domain
    "http://busbylabs.com"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the trained neural network
net = Network([784, 56, 28, 14, 10, 1])
net.load("nn.dat")  # Load the saved weights and biases

# Define a Pydantic model for the request body
class ImageData(BaseModel):
    image: list  # Expecting a list of pixel values

# Define an endpoint for digit prediction
@app.post("/predict")
async def predict(data: ImageData):
    # Validate and process the image data
    if not data.image:
        raise HTTPException(status_code=400, detail="No image data provided")

    # Convert the image data to a NumPy array
    image = np.array(data.image).reshape(784, 1) / 255.0  # Normalize to [0, 1]

    # Feed the image into the network
    output = net.feedforward(image)
    predicted_digit = int(np.argmax(output))  # Get the predicted digit

    # Return the prediction as JSON
    return {"digit": predicted_digit}