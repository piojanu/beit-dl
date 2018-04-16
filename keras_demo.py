from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical

from sklearn import datasets
from sklearn.cross_validation import train_test_split

import matplotlib.pyplot as plt
import numpy as np

# Prepare data
digits = datasets.load_digits()
# NOTE: Explain what is stratified division and why we should use it
data_train, data_test, target_train, target_test = \
  train_test_split(digits.data, digits.target, test_size=0.2, random_state=7, stratify=digits.target)
_, n_features = digits.data.shape

# Visualize data
IMAGE_SHAPE = (8, 8)
for i, image in enumerate(data_train[:20, :]):
    plt.subplot(5, 4, i + 1)
    plt.imshow(image.reshape(IMAGE_SHAPE), cmap=plt.cm.gray)
plt.show()
    
# Prepare targets
onehot_train = to_categorical(target_train) # NOTE: Explain what is onehot vector...
onehot_test = to_categorical(target_test)   #       ...and why we use it.

# Build model
model = Sequential() # NOTE: Show doc for Sequential, Dense etc. and explain what we use
model.add(Dense(100, activation='relu', input_shape=(n_features,))) # NOTE: Explain what is and why use ReLU
model.add(Dense(10, activation='softmax')) # NOTE: Explain what is and why use softmax

# Compile model
model.compile( # NOTE: Show compile docs and explain what we use
    loss='categorical_crossentropy', # NOTE: Explain what is crossentropy and tell about it's alternative SVM
    optimizer='sgd', # NOTE: Explain what is SGD, show on the table
    metrics=['accuracy'] # NOTE: We add metric, because raw loss is hard to interpret
)

# Fit model
# NOTE: Show fit docs and explain what we use and what it does
model.fit(data_train, onehot_train, epochs=25, batch_size=64, validation_split=0.1)

# Visualize weights
# NOTE: You can't see anything, images are too small
layer = np.transpose(model.get_weights()[0])
for i, image in enumerate(layer[:20, :]):
    plt.subplot(5, 4, i + 1)
    plt.imshow(image.reshape(IMAGE_SHAPE), cmap=plt.cm.gray)
plt.show()

# Evaluate model
# NOTE: Show evaluate docs and explain what we use and what it does
results = model.evaluate(data_test, onehot_test, batch_size=len(data_test), verbose=0)
print("\n[!] Evaluation results:")
print("{0}: {2:.3f}, {1}: {3:.3f}".format(*model.metrics_names, *results))
