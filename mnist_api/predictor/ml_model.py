import numpy as np
import os
from PIL import Image
import cv2
import base64
from io import BytesIO


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
    probs = softmax(z2) 

    top3_indices = probs[0].argsort()[::-1][:3] 
    top3_probs = probs[0, top3_indices]

    
    return [(int(digit), float(prob)) for digit, prob in zip(top3_indices, top3_probs)]


def center_and_pad(img, size=28):
 
    _, img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)

  
    ys, xs = np.where(img > 0)
    if len(xs) == 0 or len(ys) == 0:
    
        digit = img
    else:
        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()
        digit = img[y_min:y_max+1, x_min:x_max+1]

   
    h, w = digit.shape
    scale = 20.0 / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    digit_resized = cv2.resize(digit, (new_w, new_h), interpolation=cv2.INTER_AREA)

  
    canvas = np.zeros((size, size), dtype=np.uint8)
    start_x = (size - new_w) // 2
    start_y = (size - new_h) // 2
    canvas[start_y:start_y+new_h, start_x:start_x+new_w] = digit_resized

    ys, xs = np.where(canvas > 0)
    if len(xs) > 0 and len(ys) > 0:
        cy, cx = ys.mean(), xs.mean()
        shiftx, shifty = int(round(size/2 - cx)), int(round(size/2 - cy))
        M = np.float32([[1, 0, shiftx], [0, 1, shifty]])
        canvas = cv2.warpAffine(canvas, M, (size, size))

   
    return canvas / 255.0


def preprocess_canvas_image(data_url):
    """
    Converts base64 React canvas image to 28x28 flattened array
    """
   
    header, encoded = data_url.split(",", 1)
    data = base64.b64decode(encoded)
    img = Image.open(BytesIO(data)).convert("L") 
    img = np.array(img)

   
    digit = center_and_pad(img, size=28)

    
    return digit.reshape(1, 784)


learning_rate = 0.01

def train_on_sample(X, y_true):
    """
    X: shape (1,784)
    y_true: int 0-9
    Updates global weights W1,b1,W2,b2 in memory only
    """

    global W1, b1, W2, b2

    z1 = X @ W1 + b1
    a1 = relu(z1)

    z2 = a1 @ W2 + b2
    probs = softmax(z2)


    y_onehot = np.zeros((1,10))
    y_onehot[0, y_true] = 1

    dz2 = probs - y_onehot
    dW2 = a1.T @ dz2
    db2 = dz2

    da1 = dz2 @ W2.T
    dz1 = da1 * (z1 > 0)

    dW1 = X.T @ dz1
    db1 = dz1

  
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2

    return probs

    
def saliency_map(X):
    """
    Returns:
        importance: pixel importance (28x28)
        probs: output probabilities
        a1: hidden layer activations
    """
    # Forward
    z1 = X @ W1 + b1
    a1 = relu(z1)
    z2 = a1 @ W2 + b2
    probs = softmax(z2)

    # Pixel importance (gradient of top class w.r.t input)
    pred_class = np.argmax(probs)
    dz2 = np.zeros_like(z2)
    dz2[0, pred_class] = 1
    da1 = dz2 @ W2.T
    dz1 = da1 * (z1 > 0)
    dX = dz1 @ W1.T
    importance = np.abs(dX)


    return importance.reshape(28,28), probs, a1
def neuron_pixel_contributions(X, top_k=3):
    """
    Returns:
        contributions: list of top_k neurons' pixel importance heatmaps
        top_neurons: indices of top_k neurons
    """
    # Forward
    z1 = X @ W1 + b1
    a1 = relu(z1)
    z2 = a1 @ W2 + b2
    probs = softmax(z2)
    predicted_class = np.argmax(probs)

    # Find top_k neurons
    top_indices = a1[0].argsort()[::-1][:top_k]

    contributions = []
    for i in top_indices:
        # Contribution of neuron i to predicted class in pixel space
        pixel_contrib = a1[0,i] * W1[:, i]  # shape: 784
        contributions.append(pixel_contrib.reshape(28,28))

    return contributions, top_indices, probs, a1
def neuron_class_contributions(X, top_k=3):
    """
    Returns per-neuron contributions to the predicted class in pixel space.

    Args:
        X: Input array of shape (1, 784)
        top_k: Number of top contributing neurons to return

    Returns:
        top_neuron_maps: list of pixel-level heatmaps (28x28) for top_k neurons
        top_neurons: indices of top_k neurons
        probs: output probabilities
        hidden_activations: hidden layer activations
        predicted_class: integer class predicted
        neuron_class_contribs: contribution of each top neuron to predicted class
    """
    # Forward pass
    z1 = X @ W1 + b1
    a1 = relu(z1)
    z2 = a1 @ W2 + b2
    probs = softmax(z2)
    predicted_class = int(np.argmax(probs))

    # Compute contribution of each hidden neuron to predicted class
    neuron_contrib_values = a1[0] * W2[:, predicted_class]  # shape: (hidden_size,)

    # Pick top_k neurons by contribution
    top_indices = neuron_contrib_values.argsort()[::-1][:top_k]

    # Map each top neuron’s contribution back to pixels
    top_neuron_maps = []
    top_neuron_class_contribs = []
    for i in top_indices:
        # Pixel-level map
        pixel_map = neuron_contrib_values[i] * W1[:, i]  # 784
        pixel_map = pixel_map.reshape(28,28)
        pixel_map = pixel_map - np.min(pixel_map)
        pixel_map = pixel_map / (np.max(pixel_map)+1e-8)
        top_neuron_maps.append(pixel_map)
        # Contribution value
        top_neuron_class_contribs.append(float(neuron_contrib_values[i]))

    return top_neuron_maps, top_indices, probs, a1, predicted_class, top_neuron_class_contribs