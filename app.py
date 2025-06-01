from flask import Flask, request, jsonify
import numpy as np
from network import Network

# Initialize Flask app
app = Flask(__name__)

# Load the trained neural network
net = Network([784, 56, 28, 14, 10, 1])
net.load("nn.dat")  # Load the saved weights and biases

# Define an endpoint for digit prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Get the image data from the request
    data = request.json
    if 'image' not in data:
        return jsonify({'error': 'No image data provided'}), 400

    # Convert the image data to a NumPy array
    image = np.array(data['image']).reshape(784, 1) / 255.0  # Normalize to [0, 1]

    # Feed the image into the network
    output = net.feedforward(image)
    predicted_digit = int(np.argmax(output))  # Get the predicted digit

    # Return the prediction as JSON
    return jsonify({'digit': predicted_digit})

# Run the Flask app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)