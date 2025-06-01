from network import Network
import tkinter as tk
from PIL import Image, ImageOps
import numpy as np

net = Network([784, 56, 10])

net.load('nn.dat')


# Function to clear the canvas
def clear_canvas():
    canvas.delete("all")


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
# Create the Tkinter window
window = tk.Tk()
window.title("Draw a Digit")

# Create a canvas for drawing
canvas = tk.Canvas(window, width=560, height=560, bg="white")  # Increased size
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

def reset_position(event):
    global last_x, last_y
    last_x, last_y = None, None


last_x, last_y = None, None

def draw(event):
    block_size = 20  # Size of each block (adjust as needed)
    brush_size = 2 * block_size
    x, y = event.x, event.y

    # Calculate the top-left corner of the block
    x1 = (x // block_size) * block_size
    y1 = (y // block_size) * block_size

    # Calculate the bottom-right corner of the block
    x2 = x1 + brush_size
    y2 = y1 + brush_size

    # Draw a rectangle (block) on the canvas
    canvas.create_rectangle(x1, y1, x2, y2, fill="black", outline="black")

canvas.bind("<B1-Motion>", draw)
canvas.bind("<ButtonRelease-1>", reset_position)


window.mainloop()