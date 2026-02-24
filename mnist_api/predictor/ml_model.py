import numpy as np
import os

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


def predict(x):   # 👈 THIS MUST EXIST
    x = x / 255.0

    z1 = x @ W1 + b1
    a1 = relu(z1)

    z2 = a1 @ W2 + b2
    probs = softmax(z2)
    pred_index = int(np.argmax(probs))
    one_hot = np.zeros(10)
    one_hot[pred_index] = 1
    return one_hot


from PIL import Image


# Load your image
img = Image.open("/mnt/data/e360ea10-1991-4acf-9d5f-9c3dc81600fb.png").convert("L")  # grayscale
img = img.resize((28, 28))  # MNIST size

# Convert to NumPy array
img_array = np.array(img)

# Invert if needed (MNIST is white digit on black)
img_array = 255 - img_array

# Flatten
img_flat = img_array.reshape(1, 784)

# Predict
one_hot_pred = predict(img_flat)  # your function
pred_index = int(np.argmax(one_hot_pred))

print("Predicted digit:", pred_index)