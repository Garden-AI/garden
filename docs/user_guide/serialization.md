# Serialization of Machine Learning Models in Python

Serialization is the process of converting an object or data structure into a format that can be easily stored or transmitted. In the context of machine learning, serialization allows us to save trained models so that they can be loaded and used later without retraining.

In this tutorial, we will demonstrate how to serialize machine learning models using toy examples in Python for three different libraries: scikit-learn, TensorFlow, and PyTorch.

## Table of Contents

1. [Serialization with scikit-learn](#serialization-with-scikit-learn)
   1.1. [Using pickle](#using-pickle)
   1.2. [Using joblib](#using-joblib)

2. [Serialization with TensorFlow](#serialization-with-tensorflow)

3. [Serialization with PyTorch](#serialization-with-pytorch)

## 1. Serialization with scikit-learn

Scikit-learn is a popular machine learning library in Python, and it provides built-in functions to serialize models using both pickle and joblib.

### 1.1. Using pickle

Pickle is a standard Python library used for serializing and deserializing Python objects. We can use it to save scikit-learn models to disk.

>[!Note]
> Example CLI registration for pickle with pkl exstention:
> `garden-ai model register path/to/model.pkl sklearn --serialize-type pickle`
>  Example CLI registration for pickle:
> `garden-ai model register path/to/model.pkl sklearn`
> (pickle is the default so serialize-type is not needed here)

```python
# Example: Serialize and deserialize a scikit-learn model using pickle

import pickle
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

# Load a toy dataset (e.g., Iris dataset)
data = load_iris()
X, y = data.data, data.target

# Create and train a scikit-learn model (e.g., Logistic Regression)
model = LogisticRegression()
model.fit(X, y)

# Serialize the model using pickle
with open('model_pickle.pkl', 'wb') as f:
    pickle.dump(model, f)

# Deserialize the model using pickle
with open('model_pickle.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

# Use the loaded model for predictions
predictions = loaded_model.predict(X)
```

### 1.2. Using joblib
Joblib is an alternative to pickle, designed to be more efficient for storing large numerical arrays. It is also supported by scikit-learn.

>[!Note]
> Example CLI registration for joblib with pkl exstention:
> `garden-ai model register path/to/model.pkl sklearn --serialize-type joblib`
>  Example CLI registration for joblib with joblib exstention::
> `garden-ai model register path/to/model.joblib sklearn --serialize-type joblib`

```python
# Example: Serialize and deserialize a scikit-learn model using joblib

import joblib
from sklearn.datasets import load_iris
from sklearn.svm import SVC

# Load a toy dataset (e.g., Iris dataset)
data = load_iris()
X, y = data.data, data.target

# Create and train a scikit-learn model (e.g., Support Vector Machine)
model = SVC()
model.fit(X, y)

# Serialize the model using joblib
joblib.dump(model, 'model_joblib.pkl')

# Deserialize the model using joblib
loaded_model = joblib.load('model_joblib.pkl')

# Use the loaded model for predictions
predictions = loaded_model.predict(X)
```

You can also use joblib files:
```python
dump(clf, 'filename.joblib')
clf = load('filename.joblib')
```

You can refer to [scikit-learn documentation's](https://scikit-learn.org/stable/model_persistence.html#python-specific-serialization) for additional examples.


## 2. Serialization with TensorFlow
Below you will find code for serializing a Tensorflow model. The location you save your model in will be the path refernced when uploading the model.
>[!Note]
> Example CLI registration for h5 extension: `garden-ai model register path/to/model.h5 tensorflow`
> Example CLI registration for default tf model: `garden-ai model register path/to/model tensorflow`


```python
# Example: Serialize and deserialize a TensorFlow model

import tensorflow as tf

# Create a toy TensorFlow model (e.g., a simple neural network)
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(10,)),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile and train the model (using random data for demonstration purposes)
model.compile(optimizer='adam', loss='binary_crossentropy')
model.fit(tf.random.normal((100, 10)), tf.random.uniform((100, 1)))

# Serialize the model
model.save('model_tensorflow')
# All 3 options listed out below:
# model.save('my_model.keras')  # the Keras v3 version
# model.save('model_tensorflow', save_format = 'tf') # save as model w/o file extension
# model.save('model_tensorflow', save_format = 'h5') # save as .h5 file

# Load the model
loaded_model = tf.keras.models.load_model('model_tensorflow')

# Use the loaded model for predictions
predictions = loaded_model.predict(tf.random.normal((10, 10)))
```

Additional documentation for saving/serializing tensorflow models can be found [in this documentation](https://www.tensorflow.org/tutorials/keras/save_and_load#save_the_entire_model).

## 3. Serialization with PyTorch

Below you will find code for serializing a PyTorch model. The location you save your model in will be the path refernced when uploading the model.

>[!Note]
> Example CLI registration: `garden-ai model register path/to/model.pth pytorch`

```python
# Example: Serialize and deserialize a PyTorch model

import torch
import torch.nn as nn

# Create a toy PyTorch model (e.g., a simple neural network)
class ToyModel(nn.Module):
    def __init__(self):
        super(TheModelClass, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Initialize model
model = ToyModel()

# Initialize optimizer
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Print model's state_dict
print("Model's state_dict:")
for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())

# Print optimizer's state_dict
print("Optimizer's state_dict:")
for var_name in optimizer.state_dict():
    print(var_name, "\t", optimizer.state_dict()[var_name])

# Serialize the model
# torch.save(model.state_dict(), PATH)
torch.save(model.state_dict(), 'model_pytorch.pth')

# Create an instance of the model and load the state_dict
loaded_model = ToyModel(*args, **kwargs)
loaded_model.load_state_dict(torch.load('model_pytorch.pth'))
loaded_model.eval()
```

Additional documentation can be found on saving/serializing PyTorch models [here](https://pytorch.org/tutorials/beginner/saving_loading_models.html).
