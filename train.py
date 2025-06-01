from network import Network
from network import vec_val
from keras.datasets import mnist
import numpy as np
import tkinter as tk
from PIL import Image, ImageOps
from time import sleep

def load_training_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train_sample = x_train / 255.0
    y_train_sample = y_train

    x_test_sample = x_test / 255.0
    y_test_sample = y_test

    print(x_train_sample)

    training_inputs = [np.reshape(x, (784, 1)) for x in x_train_sample]
    training_values = [vec_val(y) for y in y_train_sample]

    training_data = list(zip(training_inputs, training_values))
    training_data_units = [None] * 10
    for i in range(0,10):
        training_data_units[i] = [(x,y) for x, y in training_data if np.argmax(y) == i]

    test_inputs = [np.reshape(x, (784, 1)) for x in x_test_sample]
    test_values = [vec_val(y) for y in y_test_sample]

    test_data = list(zip(test_inputs, test_values))

    print(training_inputs[1])

    print("Training inputs successfully loaded!")

    return training_data, test_data

net = Network([784, 56, 10])

vertical = np.array([(np.append(np.append(np.zeros((1, 28*i)), np.ones((1, 28))), np.zeros((1, 784-(28*(i+1)))))) for i in range (0, 28)])
horizontal = np.array([np.insert(np.zeros((1, 27)), i, 1) for i in range(28)])
horizontal = np.tile(horizontal, 28)
print(np.shape(vertical))
print(np.shape(horizontal))
#net.weights[0] = np.reshape(np.append(vertical, horizontal), (56, 784))
#net.load('nn.dat')


training_data, test_data = load_training_data()

#net.evaluate(test_data)
#sleep(50)
net.SGD(10, 5, 0.05, training_data, test_data)
net.save("nn.dat")


def predict_digit():
    # Save the canvas content as an image
    canvas.postscript(file="digit.eps")
    img = Image.open("digit.eps").convert("L")  # Convert to grayscale
    img = ImageOps.invert(img)  # Invert colors (white background, black digit)
    img = img.resize((28, 28))  # Resize to 28x28 pixels
    img_array = np.array(img) / 255.0  # Normalize pixel values to [0, 1]
    img_array = img_array.reshape(784, 1)  # Flatten to (784, 1)

    # Feed the image into the network
    output = net.feedforward(img_array)
    predicted_digit = np.argmax(output)  # Get the digit with the highest probability

    # Display the result
    result_label.config(text=f"Predicted Digit: {predicted_digit}")

# Function to clear the canvas
def clear_canvas():
    canvas.delete("all")

# Create the Tkinter window
window = tk.Tk()
window.title("Draw a Digit")

# Create a canvas for drawing
canvas = tk.Canvas(window, width=280, height=280, bg="white")
canvas.grid(row=0, column=0, columnspan=2)

# Add buttons for prediction and clearing the canvas
predict_button = tk.Button(window, text="Predict", command=predict_digit)
predict_button.grid(row=1, column=0)

clear_button = tk.Button(window, text="Clear", command=clear_canvas)
clear_button.grid(row=1, column=1)

# Add a label to display the prediction result
result_label = tk.Label(window, text="Draw a digit and click Predict")
result_label.grid(row=2, column=0, columnspan=2)

# Enable drawing on the canvas
def draw(event):
    x, y = event.x, event.y
    canvas.create_oval(x-10, y-10, x+10, y+10, fill="black", width=0)

canvas.bind("<B1-Motion>", draw)

# Run the Tkinter event loop
window.mainloop()

#net.feedforward(training_inputs[0])
