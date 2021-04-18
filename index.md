## Introduction

We have performed the basic neural network in PyTorch. tensorFlow, in another way, is a framework for creating machine learning models as well as PyTorch. We will still train the penguin dataset through using TensorFlow in this article. 

Our goal is to classify penguins into species(3 classes) based on the length and depth of the bill, the flipper length and the body mass. The data set used is a subset of data collected and made available by Dr. Kristen Gorman and the Palmer Station, Antarctica LTER

## 1. Exploring the dataset

Before using TensorFlow to create a model, we should load the data from the Palmer Islands penguins dataset.

```python
import pandas as pd

# load the training dataset (excluding rows with null values)
penguins = pd.read_csv('data/penguins.csv').dropna()

# Deep Learning models work best when features are on similar scales
# In a real solution, we'd implement some custom normalization for each feature, but to keep things simple
# we'll just rescale the FlipperLength and BodyMass so they're on a similar scale to the bill measurements
penguins['FlipperLength'] = penguins['FlipperLength']/10
penguins['BodyMass'] = penguins['BodyMass']/100

# The dataset is too small to be useful for deep learning
# So we'll oversample it to increase its size
for i in range(1,3):
    penguins = penguins.append(penguins)

# Display a random sample of 10 observations
sample = penguins.sample(10)
sample
```

The result:

![image](https://user-images.githubusercontent.com/71245576/115160336-b7292480-a065-11eb-90b6-ada0484eace6.png)

The Species column is the label to predict, each label value represents a class of penguin species, encoded as 0,1, or 2.

Now let's see the actual species corresponding to the encodding numbers:
```python
penguin_classes = ['Adelie', 'Gentoo', 'Chinstrap']
print(sample.columns[0:5].values, 'SpeciesName')
for index, row in penguins.sample(10).iterrows():
    print('[',row[0], row[1], row[2],row[3], int(row[4]), ']',penguin_classes[int(row[-1])])
```

Gentoo, Adelie and Chinstrap are the species of the label.

Now, separate the features and the label as well as splitting the data into valiation dataset and training dataset.

```python
from sklearn.model_selection import train_test_split

features = ['CulmenLength','CulmenDepth','FlipperLength','BodyMass']
label = 'Species'
   
# Split data 70%-30% into training set and test set
x_train, x_test, y_train, y_test = train_test_split(penguins[features].values,
                                                    penguins[label].values,
                                                    test_size=0.30,
                                                    random_state=0)

print ('Training Set: %d, Test Set: %d \n' % (len(x_train), len(x_test)))
print("Sample of features and labels:")

# Take a look at the first 25 training features and corresponding labels
for n in range(0,24):
    print(x_train[n], y_train[n], '(' + penguin_classes[y_train[n]] + ')')
```
The number of observations in training set is 957, and 411 observations in test set.

## 2. Installing and importing TensorFlow

Before using TensorFlow, we need to run commands to install and import the libraries we intend to use:

```python
!pip install --upgrade tensorflow
```

It takes a few minutes in Azure:

![image](https://user-images.githubusercontent.com/71245576/115160527-a9c06a00-a066-11eb-8bec-2adddfe5b5c3.png)

Now import tensorflow into the platform:

```python
import tensorflow
from tensorflow import keras
from tensorflow.keras import models
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import utils
from tensorflow.keras import optimizers

# Set random seed for reproducability
tensorflow.random.set_seed(0)

print("Libraries imported.")
print('Keras version:',keras.__version__)
print('TensorFlow version:',tensorflow.__version__)
```

## 3. Neural network modeling

Before the modeling, we need to prepare the data for TensorFlow. We have already loaded our data and split it into training and validation datasets, however, we need to do the same for TensorFlow, to set the data type of our features to 32-bit floatingpoint numbers and specify that the labels represent categorical classes rather than numeric values.

```python
# Set data types for float features
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# Set data types for categorical labels
y_train = utils.to_categorical(y_train)
y_test = utils.to_categorical(y_test)
print('Ready...')
```
It is ready.

Now let's define a neural network, we are going to create a network that consists of 3 fully connected layers.

```python
# Define a classifier network
hl = 10 # Number of hidden layer nodes

model = Sequential()
model.add(Dense(hl, input_dim=len(features), activation='relu'))
model.add(Dense(hl, input_dim=hl, activation='relu'))
model.add(Dense(len(penguin_classes), input_dim=hl, activation='softmax'))

print(model.summary())
```

Train the model now:

```python
#hyper-parameters for optimizer
learning_rate = 0.001
opt = optimizers.Adam(lr=learning_rate)

model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

# Train the model over 50 epochs using 10-observation batches and using the test holdout dataset for validation
num_epochs = 50
history = model.fit(x_train, y_train, epochs=num_epochs, batch_size=10, validation_data=(x_test, y_test))
```

We trained for 50 epoches, the final accuracy is 0.9708.

![image](https://user-images.githubusercontent.com/71245576/115160745-e6d92c00-a067-11eb-8679-5bd27354f899.png)

Review the training and validation loss now, hopefully we want the training loss and validation loss follow a similar trend:

```python
%matplotlib inline
from matplotlib import pyplot as plt

epoch_nums = range(1,num_epochs+1)
training_loss = history.history["loss"]
validation_loss = history.history["val_loss"]
plt.plot(epoch_nums, training_loss)
plt.plot(epoch_nums, validation_loss)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['training', 'validation'], loc='upper right')
plt.show()
```

It seems good:

![image](https://user-images.githubusercontent.com/71245576/115160789-26077d00-a068-11eb-947f-caeadcbfddc1.png)

Let's view the learned weights and biases for each layers:

![image](https://user-images.githubusercontent.com/71245576/115160819-40d9f180-a068-11eb-9660-fe9a67190a51.png)

![image](https://user-images.githubusercontent.com/71245576/115160828-4cc5b380-a068-11eb-87ca-ce60e399fb7e.png)

Until now we can consider the model performs pretty well right? But it is typically useful to dig a little deeper and compare the predictions for each possible class. A common way to visualize the performance of a classification model is to create a confusion matrix:

```python
# Tensorflow doesn't have a built-in confusion matrix metric, so we'll use SciKit-Learn
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
%matplotlib inline


class_probabilities = model.predict(x_test)
predictions = np.argmax(class_probabilities, axis=1)
true_labels = np.argmax(y_test, axis=1)

# Plot the confusion matrix
cm = confusion_matrix(true_labels, predictions)
plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
plt.colorbar()
tick_marks = np.arange(len(penguin_classes))
plt.xticks(tick_marks, penguin_classes, rotation=85)
plt.yticks(tick_marks, penguin_classes)
plt.xlabel("Actual Class")
plt.ylabel("Predicted Class")
plt.show()
```

The confusion matrix show a strong diagonal line indicating that there are more correct than incorrect predictions for each classes:

![image](https://user-images.githubusercontent.com/71245576/115160908-9d3d1100-a068-11eb-941c-d14ab1e1d4f4.png)

Save the trained model now:

```python
# Save the trained model
modelFileName = 'models/penguin-classifier.h5'
model.save(modelFileName)
del model  # deletes the existing model variable
print('model saved as', modelFileName)
```

Use it to predict classes for a new penguin observation:
```python
# Load the saved model
model = models.load_model(modelFileName)

# CReate a new array of features
x_new = np.array([[50.4,15.3,20,50]])
print ('New sample: {}'.format(x_new))

# Use the model to predict the class
class_probabilities = model.predict(x_new)
predictions = np.argmax(class_probabilities, axis=1)

print(penguin_classes[predictions[0]])
```
The species of this observation is Gentoo.

## Reference:

Train and evaluate deep learning models, retrieved from https://docs.microsoft.com/en-us/learn/modules/train-evaluate-deep-learn-models/




