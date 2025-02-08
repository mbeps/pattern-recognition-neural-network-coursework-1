# Assessed Coursework: Deep Discriminant Neural Networks
This part of the coursework requires you to build and train a deep neural network for classifying handwritten digits.
Your network must be built using KERAS
Please see the module’s KEATS page for a tutorial on using KERAS
This tutorial describes how to build networks for classifying the MNIST dataset.

This dataset consists of 70000 28-by-28 pixel images of handwritten digits, which are by convention split into 60000 images for training and 10000 for testing
The tutorial provides instructions on how to build both a MLP and a CNN for MNIST digit classification
You can use either of these as a starting point and experiment with making modifications to improve performance
You may also use pre-built networks from the model zoo, or you may choose to build your own deep network based on your own or someone else’s design
Please note that in all cases you must train your own neural network: anyone who is found to have submitted a model that they have not trained themselves (e.g
a pre-trained network obtained from the internet) will receive a mark of zero.

Your trained neural network must be saved as a .h5 file and submitted via KEATS
You can give this file any name you wish, but it must end with a .h5 extension
Note it should be possible to test your neural network to predict class labels on a standard PC running Linux without the need for special hardware, such as a GPU or RAM in excess of 16GB
16GB is sufficient to test a model with around 1 billion parameters, however, KEATS places a limit of 500MB on the size of any file that you can upload which limits the size of any network you can submit to around 40 to 50 million parameters.

Your neural network will be assessed by testing the accuracy with which it classifies another, unseen, test data (“my testset”)
This testing data consist of about 12000 samples and is distinct from the testing and training data provided as part of the MNIST dataset, but in common with MNIST (after applying the pre-processing steps described in the tutorial) consists of images of handwritten digits that: have pixel values between 0 and 1, large values correspond to pen strokes while small values correspond to the background (i.e
images contain white digits on a black background when shown with a gray colormap), have been scaled and cropped (but not otherwise augmented) to be 28-by-28 pixel images with the digits approximately in the centre
The primary difference is that the digits in my testset have been written by different people, so may have different characteristics.

The reasons for evaluating performance on this new dataset are two fold
Firstly, because the MNIST test set is freely available it is not possible to ensure that submitted networks have not been trained with this data
If a network is assessed with the same data it was trained on, this does not provide a fair assessment of how well the network can classify unseen data
Secondly, this scenario is more consistent with the sort of problem you might be faced with in the real-world, where you need to develop a system that will work “in production”, with new data that may not have even been generated yet
It is difficult to know if a classifier will generalise well to unseen data, but to have a chance of succeeding it is important for you to ensure that your model is not over-fitted to the MNIST data and that it can can cope with digits that may vary in appearance compared to the MNIST data.

Marks will be awarded based on how accurately your neural network performs the classification of my testset
Note that the tutorial on using KERAS explains how to build two networks for classifying the MNIST data: the MLP will classify the standard MNIST test set with an accuracy of about 98% and will classify my testset with an accuracy of about 90%; the CNN will classify the standard MNIST test set with an accuracy of about 99% and will classify my testset with an accuracy of about 95%
Hence, 95% accuracy is considered the baseline and will earn 25% of the available marks
Higher marks will be awarded for accuracies greater than 95%
Exactly how performance maps onto a grade will be decided once all the submissions have been marked.

To help you check that your network is compatible with my test environment, here is the code I will be using to test its performance on the MNIST dataset:

```py
import numpy as np
import os
from tensorflow.keras.models import load_model

# load .h5 file of arbitrary name for testing (last if more than one)
print(os.getcwd())

for file in os.listdir(os.getcwd()):
    if file.endswith(".h5"):
        print(file)
        net = load_model(file)

net.summary()

# determine what type of network this is
conf = net.layers[0].get_config()
inshape = conf['batch_input_shape']

if inshape[1] == 28:
    netType = 'CNN'
else:
    netType = 'MLP'

# test with MNIST data
from tensorflow.keras.datasets import mnist

(x_train, labels_train), (x_test, labels_test) = mnist.load_data()

x_test = x_test.astype('float32')
x_test /= 255

if netType in ['MLP']:
    x_test = x_test.reshape(10000, 784)
else:
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

outputs = net.predict(x_test)
labels_predicted = np.argmax(outputs, axis=1)

correct_classified = sum(labels_predicted == labels_test)
print('Percentage correctly classified MNIST= ', 100 * correct_classified / labels_test.size)
```