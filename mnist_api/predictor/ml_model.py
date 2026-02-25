import numpy as np
import os
from PIL import Image
import cv2
import base64
from io import BytesIO

# Load model weights
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "mnist_xai_2500.npz")
data = np.load(model_path)

W1 = data["W1"]
b1 = data["b1"]
W2 = data["W2"]
b2 = data["b2"]

def relu(x):
    return np.maximum(0, x)

def softmax(x):
    e = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e / np.sum(e, axis=1, keepdims=True)

def predict(X):
    """
    X: numpy array of shape (1, 784), values normalized 0-1
    Returns: integer 0-9
    """
    z1 = X @ W1 + b1
    a1 = relu(z1)
    z2 = a1 @ W2 + b2
    probs = softmax(z2)
    return int(np.argmax(probs))


def center_and_pad(img, size=28):
    """
    Centers the digit using center-of-mass and pads to maintain aspect ratio.
    """
    # Threshold to binary
    _, img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        x, y, w, h = cv2.boundingRect(contours[0])
        digit = img[y:y+h, x:x+w]
    else:
        digit = img  # empty canvas

    # Resize while keeping aspect ratio
    h, w = digit.shape
    scale = 20.0 / max(h, w)  # scale largest dimension to 20 pixels
    new_w = int(w * scale)
    new_h = int(h * scale)
    digit_resized = cv2.resize(digit, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Create 28x28 blank image
    canvas = np.zeros((size, size), dtype=np.uint8)

    # Compute top-left coordinates to paste resized digit
    start_x = (size - new_w) // 2
    start_y = (size - new_h) // 2
    canvas[start_y:start_y+new_h, start_x:start_x+new_w] = digit_resized

    # Center-of-mass shift
    cy, cx = np.array(np.where(canvas > 0)).mean(axis=1)
    shiftx = np.round(size/2 - cx).astype(int)
    shifty = np.round(size/2 - cy).astype(int)
    M = np.float32([[1, 0, shiftx], [0, 1, shifty]])
    centered = cv2.warpAffine(canvas, M, (size, size))

    # Normalize
    centered = centered / 255.0
    return centered


def preprocess_canvas_image(data_url):
    """
    Converts base64 React canvas image to 28x28 flattened array
    """
    # Decode base64 to PIL Image
    header, encoded = data_url.split(",", 1)
    data = base64.b64decode(encoded)
    img = Image.open(BytesIO(data)).convert("L")  # grayscale
    img = np.array(img)

    # Center, pad, and normalize
    digit = center_and_pad(img, size=28)

    # Flatten
    return digit.reshape(1, 784)