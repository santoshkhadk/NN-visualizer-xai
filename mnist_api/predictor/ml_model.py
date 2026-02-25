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

def predict_top3(X):
    """
    X: numpy array of shape (1, 784), values normalized 0-1
    Returns: list of top 3 (digit, probability) tuples
    """
    z1 = X @ W1 + b1
    a1 = relu(z1)
    z2 = a1 @ W2 + b2
    probs = softmax(z2)  # shape (1, 10)

    top3_indices = probs[0].argsort()[::-1][:3]  # indices of top 3
    top3_probs = probs[0, top3_indices]

    # Return list of (digit, probability)
    return [(int(digit), float(prob)) for digit, prob in zip(top3_indices, top3_probs)]


def center_and_pad(img, size=28):
    # Threshold to binary
    _, img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)

    # Find all non-zero pixels
    ys, xs = np.where(img > 0)
    if len(xs) == 0 or len(ys) == 0:
        # Empty image
        digit = img
    else:
        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()
        digit = img[y_min:y_max+1, x_min:x_max+1]

    # Resize while keeping aspect ratio
    h, w = digit.shape
    scale = 20.0 / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    digit_resized = cv2.resize(digit, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Create canvas
    canvas = np.zeros((size, size), dtype=np.uint8)
    start_x = (size - new_w) // 2
    start_y = (size - new_h) // 2
    canvas[start_y:start_y+new_h, start_x:start_x+new_w] = digit_resized

    # Center using center-of-mass
    ys, xs = np.where(canvas > 0)
    if len(xs) > 0 and len(ys) > 0:
        cy, cx = ys.mean(), xs.mean()
        shiftx, shifty = int(round(size/2 - cx)), int(round(size/2 - cy))
        M = np.float32([[1, 0, shiftx], [0, 1, shifty]])
        canvas = cv2.warpAffine(canvas, M, (size, size))

    # Normalize
    return canvas / 255.0


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