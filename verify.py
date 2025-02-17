import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import mnist

# Print the current directory
print("Current directory:", os.getcwd())

# Find and load the last .h5 model in the directory
model_file = None
for file in os.listdir(os.getcwd()):
    if file.endswith(".h5"):
        print("Found model:", file)
        model_file = file

if model_file is None:
    raise FileNotFoundError("No .h5 file found in the current directory.")

net = load_model(model_file)
net.summary()

# Determine the network type using the model's input shape
inshape = net.input_shape
print("Model input shape:", inshape)

if len(inshape) == 4:
    netType = 'CNN'
elif len(inshape) == 2:
    netType = 'MLP'
else:
    raise ValueError("Unsupported network input shape.")

print("Detected network type:", netType)

# Load MNIST data for testing
(x_train, labels_train), (x_test, labels_test) = mnist.load_data()
x_test = x_test.astype('float32') / 255.0

# Reshape the test data based on the network type
if netType == 'MLP':
    x_test = x_test.reshape(x_test.shape[0], -1)
else:  # CNN
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

# Make predictions on the test data
outputs = net.predict(x_test)
labels_predicted = np.argmax(outputs, axis=1)

# Calculate and print the percentage of correctly classified samples
correct_classified = np.sum(labels_predicted == labels_test)
accuracy = 100 * correct_classified / labels_test.size
print('Percentage correctly classified MNIST =', accuracy)
